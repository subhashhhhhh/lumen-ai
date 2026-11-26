package com.subhashh.lumen.ai

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Smart Search's index for one model: which library items are done (at which version) and their
 * vectors. A photo has one vector; a video one per kept frame, with the frame's time. One file
 * per model, in app-private storage that isn't backed up, since vectors from different models
 * can't be mixed and are cheap to rebuild.
 */
@LumenAiEngine
class EmbeddingStore private constructor(context: Context, private val model: AiModel) :
    SQLiteOpenHelper(context, file(context, model).path, null, 1) {

    init {
        setWriteAheadLoggingEnabled(true)
    }

    /** Bumped on every write, so a search cache knows its copy is stale. */
    @Volatile
    var version = 0
        private set

    override fun onCreate(db: SQLiteDatabase) {
        // frames = 0: tried and unreadable (corrupt, DRM); kept so it isn't retried every run.
        db.execSQL("CREATE TABLE items (media_id INTEGER PRIMARY KEY, modified INTEGER NOT NULL, frames INTEGER NOT NULL)")
        db.execSQL("CREATE TABLE vectors (media_id INTEGER NOT NULL, at_ms INTEGER NOT NULL, vec BLOB NOT NULL)")
        db.execSQL("CREATE INDEX vectors_media ON vectors (media_id)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit

    /** Item id → the `modified` it was indexed at. */
    fun indexed(): Map<Long, Long> = HashMap<Long, Long>().apply {
        // One transaction, as in loadAll: a big library's list spans several cursor windows.
        val db = writableDatabase
        db.beginTransactionNonExclusive()
        try {
            db.rawQuery("SELECT media_id, modified FROM items", null).use { c ->
                while (c.moveToNext()) put(c.getLong(0), c.getLong(1))
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun count(): Int = readableDatabase.rawQuery("SELECT COUNT(*) FROM items", null).use { c ->
        if (c.moveToFirst()) c.getInt(0) else 0
    }

    /** Replaces whatever [mediaId] had. */
    fun put(mediaId: Long, modified: Long, frames: List<Pair<Long, FloatArray>>) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            db.delete("vectors", "media_id = ?", arrayOf(mediaId.toString()))
            for ((at, vector) in frames) {
                db.insert("vectors", null, ContentValues().apply {
                    put("media_id", mediaId)
                    put("at_ms", at)
                    put("vec", toBytes(vector))
                })
            }
            db.insertWithOnConflict("items", null, ContentValues().apply {
                put("media_id", mediaId)
                put("modified", modified)
                put("frames", frames.size)
            }, SQLiteDatabase.CONFLICT_REPLACE)
            db.setTransactionSuccessful()
            changed(longArrayOf(mediaId))
        } finally {
            db.endTransaction()
        }
    }

    fun remove(mediaIds: Collection<Long>) {
        if (mediaIds.isEmpty()) return
        val db = writableDatabase
        db.beginTransaction()
        try {
            for (id in mediaIds) {
                val args = arrayOf(id.toString())
                db.delete("vectors", "media_id = ?", args)
                db.delete("items", "media_id = ?", args)
            }
            db.setTransactionSuccessful()
            changed(mediaIds.toLongArray())
        } finally {
            db.endTransaction()
        }
    }

    /** One item's vectors: a photo's one, or a video's frames. */
    fun vectorsOf(mediaId: Long): List<FloatArray> = ArrayList<FloatArray>().apply {
        val dim = model.spec.dim
        readableDatabase.rawQuery("SELECT vec FROM vectors WHERE media_id = ?", arrayOf(mediaId.toString())).use { c ->
            while (c.moveToNext()) {
                add(FloatArray(dim).also { ByteBuffer.wrap(c.getBlob(0)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) })
            }
        }
    }

    /** Which items recent writes touched, so a copy in memory can catch up ([changedSince]). */
    private val changes = ChangeLog(ChangeLogSize)

    @Synchronized
    private fun changed(mediaIds: LongArray) {
        version = changes.add(mediaIds)
    }

    /**
     * Every item written or removed after [since], or null when that's too long ago to tell
     * (or [since] is from before this process started): read everything again then.
     */
    @Synchronized
    fun changedSince(since: Int): Set<Long>? = changes.since(since)

    /**
     * What [loadAll] read last, while something (Smart Search) still holds it: one copy in memory
     * instead of one per feature.
     */
    @Volatile private var shared = java.lang.ref.WeakReference<Embeddings>(null)

    /** The vectors [loadAll] read last, or a fresh read: for work that doesn't need the latest items. */
    fun recent(): Embeddings = shared.get() ?: loadAll()

    /** Everything, for searching, with room to grow ([Embeddings.withRoomFor]). */
    fun loadAll(): Embeddings {
        shared.clear() // the old copy can go before the new one is read
        val dim = model.spec.dim
        // In one transaction: a cursor this big is read in windows, each a fresh query, and rows
        // removed in between (a photo deleted while indexing ran) made the next window come up
        // short ("Couldn't read row 224, col 2 from CursorWindow", a crash). Writers wait a moment.
        val db = writableDatabase
        db.beginTransactionNonExclusive()
        try {
            db.rawQuery("SELECT media_id, at_ms, vec FROM vectors ORDER BY media_id", null).use { c ->
                val all = Embeddings.withRoomFor(c.count, dim)
                val row = FloatArray(dim)
                while (c.moveToNext()) {
                    ByteBuffer.wrap(c.getBlob(2)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(row)
                    if (!all.append(c.getLong(0), c.getLong(1), row)) break // can't happen: sized from the count
                }
                db.setTransactionSuccessful()
                return all.also { shared = java.lang.ref.WeakReference(it) }
            }
        } finally {
            db.endTransaction()
        }
    }

    /**
     * Brings [all] up to date for the items in [changed]: their old rows go, their current ones
     * are appended. False when it can't (no room, or mostly dead rows): [loadAll] instead.
     */
    fun update(all: Embeddings, changed: Set<Long>): Boolean {
        if (changed.isEmpty()) return true
        all.remove(changed)
        val dim = model.spec.dim
        val row = FloatArray(dim)
        for (chunk in changed.chunked(500)) {
            val args = chunk.joinToString(",") // ids are numbers, never user text
            readableDatabase.rawQuery("SELECT media_id, at_ms, vec FROM vectors WHERE media_id IN ($args) ORDER BY media_id", null).use { c ->
                while (c.moveToNext()) {
                    ByteBuffer.wrap(c.getBlob(2)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(row)
                    if (!all.append(c.getLong(0), c.getLong(1), row)) return false
                }
            }
        }
        return all.dead <= all.size / 3
    }

    companion object {
        /** Writes [changedSince] remembers: a few hours of indexing, a few hundred KB at most. */
        private const val ChangeLogSize = 20_000

        private val open = HashMap<String, EmbeddingStore>()

        /** One open helper per model for the whole process (it's thread-safe). */
        @Synchronized
        fun of(context: Context, model: AiModel): EmbeddingStore =
            open.getOrPut(model.id) { EmbeddingStore(context.applicationContext, model) }

        /** Drops a model's index: when it's removed or replaced by another tier. */
        @Synchronized
        fun delete(context: Context, model: AiModel) {
            open.remove(model.id)?.close()
            SQLiteDatabase.deleteDatabase(file(context, model))
        }

        private fun file(context: Context, model: AiModel) =
            File(context.noBackupFilesDir, "search/${model.id}.db").apply { parentFile?.mkdirs() }

        private fun toBytes(vector: FloatArray): ByteArray =
            ByteBuffer.allocate(vector.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(vector) }.array()
    }
}
