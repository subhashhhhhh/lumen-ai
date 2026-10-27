package com.subhashh.lumen.ai

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** One face in one photo: where (fractions of the photo's width and height), its fingerprint, whose. */
class Face(
    val id: Long,
    val mediaId: Long,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val vector: FloatArray,
    val person: Long?,
    /** The person the user said this isn't (Not This Person): grouping never puts it back there. */
    val notPerson: Long?,
    /** How clearly the recognizer saw it ([FaceRecognizer.minQuality]); null if it doesn't say. */
    val quality: Float?,
)

/**
 * A person as the People row and their album show them: name (null until named), the photos
 * they're in, and the face to show for them (their biggest, so the sharpest).
 */
/** One face in a photo or video: its person (null while it waits for one), box (fractions of the frame) and moment (videos). */
data class FaceIn(val id: Long, val person: Long?, val box: List<Float>, val atMs: Long?)

data class Person(
    val id: Long,
    val name: String?,
    val hidden: Boolean,
    val mediaIds: Set<Long>,
    val faceCount: Int,
    val coverMediaId: Long,
    /** The cover face's box, fractions of its photo: left, top, right, bottom. */
    val coverBox: List<Float>,
    val coverFaceId: Long,
    /** The cover is a video's face: shown from its saved crop ([FaceStore.crop]), not the video's thumbnail. */
    val coverInVideo: Boolean,
)

/** A face to save: box as fractions, fingerprint, and for a video the moment it's from. */
class NewFace(val box: FloatArray, val vector: FloatArray, val quality: Float, val atMs: Long? = null)

/**
 * Faces found in photos and the people they're grouped into. Same lifecycle as [EmbeddingStore]:
 * app-private, not backed up. People carry what the user decides (names, merges, hidden), so
 * grouping only ever adds faces to people; it never moves a face that already has one.
 */
@LumenAiEngine
class FaceStore private constructor(context: Context) : SQLiteOpenHelper(context, file(context).path, null, 6) {
    private val crops = cropDir(context)

    init {
        setWriteAheadLoggingEnabled(true)
    }

    /** Bumped on every write, so screens know to reload. */
    var version: Int
        get() = changes.value
        private set(value) {
            changes.value = value
        }

    override fun onCreate(db: SQLiteDatabase) {
        // A photo with no faces keeps its row in items, so it isn't looked at again.
        db.execSQL("CREATE TABLE items (media_id INTEGER PRIMARY KEY, modified INTEGER NOT NULL)")
        db.execSQL(
            "CREATE TABLE faces (id INTEGER PRIMARY KEY, media_id INTEGER NOT NULL, " +
                "l REAL NOT NULL, t REAL NOT NULL, r REAL NOT NULL, b REAL NOT NULL, vec BLOB NOT NULL, person INTEGER, " +
                "not_person INTEGER, at_ms INTEGER, model TEXT, quality REAL)",
        )
        db.execSQL("CREATE INDEX faces_media ON faces (media_id)")
        db.execSQL("CREATE INDEX faces_person ON faces (person)")
        db.execSQL("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, hidden INTEGER NOT NULL DEFAULT 0, cover_face INTEGER)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // 2: Not This Person.
        if (oldVersion < 2) db.execSQL("ALTER TABLE faces ADD COLUMN not_person INTEGER")
        // 3: faces in videos, with the moment (null for photos).
        if (oldVersion < 3) db.execSQL("ALTER TABLE faces ADD COLUMN at_ms INTEGER")
        // 4: covers the user chose (Make Person Cover).
        if (oldVersion < 4) db.execSQL("ALTER TABLE people ADD COLUMN cover_face INTEGER")
        // 5: which recognizer made each face's numbers (null: SFace, before 0.3).
        if (oldVersion < 5) db.execSQL("ALTER TABLE faces ADD COLUMN model TEXT")
        // 6: how clearly the recognizer saw each face. Development builds moved faces to AdaFace
        // without it at version 5: they're read again.
        if (oldVersion < 6) {
            db.execSQL("ALTER TABLE faces ADD COLUMN quality REAL")
            if (oldVersion == 5) db.execSQL("UPDATE faces SET model = NULL")
        }
    }

    /** Item id → the `modified` it was looked at. */
    fun indexed(): Map<Long, Long> = HashMap<Long, Long>().apply {
        readableDatabase.rawQuery("SELECT media_id, modified FROM items", null).use { c ->
            while (c.moveToNext()) put(c.getLong(0), c.getLong(1))
        }
    }

    fun count(): Int = readableDatabase.rawQuery("SELECT COUNT(*) FROM items", null).use { c ->
        if (c.moveToFirst()) c.getInt(0) else 0
    }

    fun peopleCount(): Int = readableDatabase.rawQuery("SELECT COUNT(*) FROM people", null).use { c ->
        if (c.moveToFirst()) c.getInt(0) else 0
    }

    /** Replaces [mediaId]'s faces (an edited photo is looked at again); returns the new faces' ids. */
    fun put(mediaId: Long, modified: Long, faces: List<NewFace>, model: String): List<Long> {
        val db = writableDatabase
        val ids = ArrayList<Long>()
        db.beginTransaction()
        try {
            dropCrops(db, "media_id = ?", arrayOf(mediaId.toString()))
            db.delete("faces", "media_id = ?", arrayOf(mediaId.toString()))
            for (face in faces) {
                val box = face.box
                ids += db.insert("faces", null, ContentValues().apply {
                    put("media_id", mediaId)
                    put("l", box[0]); put("t", box[1]); put("r", box[2]); put("b", box[3])
                    put("vec", toBytes(face.vector))
                    put("model", model)
                    put("quality", face.quality)
                    face.atMs?.let { put("at_ms", it) }
                })
            }
            db.insertWithOnConflict("items", null, ContentValues().apply {
                put("media_id", mediaId)
                put("modified", modified)
            }, SQLiteDatabase.CONFLICT_REPLACE)
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        version++
        return ids
    }

    /** Where a video face's crop is saved (photos' faces are cut from the photo when shown). */
    fun crop(faceId: Long) = File(crops, "$faceId.jpg")

    private fun dropCrops(db: SQLiteDatabase, where: String, args: Array<String>) {
        db.rawQuery("SELECT id FROM faces WHERE at_ms IS NOT NULL AND $where", args).use { c ->
            while (c.moveToNext()) crop(c.getLong(0)).delete()
        }
    }

    fun remove(ids: Collection<Long>) {
        if (ids.isEmpty()) return
        val db = writableDatabase
        db.beginTransaction()
        try {
            for (id in ids) {
                dropCrops(db, "media_id = ?", arrayOf(id.toString()))
                db.delete("faces", "media_id = ?", arrayOf(id.toString()))
                db.delete("items", "media_id = ?", arrayOf(id.toString()))
            }
            // People whose last face went with those photos.
            db.execSQL("DELETE FROM people WHERE id NOT IN (SELECT DISTINCT person FROM faces WHERE person IS NOT NULL)")
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        version++
    }

    /** Every face whose numbers [model] made, newest row last. */
    fun faces(model: String): List<Face> = ArrayList<Face>().apply {
        readableDatabase.rawQuery("SELECT id, media_id, l, t, r, b, vec, person, not_person, quality FROM faces WHERE model = ? ORDER BY id", arrayOf(model)).use { c ->
            while (c.moveToNext()) {
                val blob = c.getBlob(6)
                val vector = FloatArray(blob.size / 4).also {
                    ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it)
                }
                add(
                    Face(
                        c.getLong(0), c.getLong(1), c.getFloat(2), c.getFloat(3), c.getFloat(4), c.getFloat(5),
                        vector, if (c.isNull(7)) null else c.getLong(7), if (c.isNull(8)) null else c.getLong(8),
                        if (c.isNull(9)) null else c.getFloat(9),
                    ),
                )
            }
        }
    }

    /** A face whose numbers another recognizer made: where to find it again. */
    class Stale(val id: Long, val mediaId: Long, val box: FloatArray, val atMs: Long?)

    /** Faces not yet moved to [model], a photo or video at a time. */
    fun stale(model: String): List<Stale> = ArrayList<Stale>().apply {
        readableDatabase.rawQuery(
            "SELECT id, media_id, l, t, r, b, at_ms FROM faces WHERE model IS NULL OR model != ? ORDER BY media_id, id", arrayOf(model),
        ).use { c ->
            while (c.moveToNext()) {
                add(Stale(c.getLong(0), c.getLong(1), floatArrayOf(c.getFloat(2), c.getFloat(3), c.getFloat(4), c.getFloat(5)), if (c.isNull(6)) null else c.getLong(6)))
            }
        }
    }

    fun count(model: String): Int = readableDatabase.rawQuery("SELECT COUNT(*) FROM faces WHERE model = ?", arrayOf(model)).use { c ->
        if (c.moveToFirst()) c.getInt(0) else 0
    }

    /**
     * Moving to a new recognizer: [found] faces get its numbers, [lost] ones (not found again)
     * go, with any person left without faces.
     */
    fun revector(found: Map<Long, FaceReader.Found>, lost: Collection<Long>, model: String) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            for ((id, face) in found) {
                db.update("faces", ContentValues().apply {
                    put("vec", toBytes(face.vector)); put("model", model); put("quality", face.quality)
                }, "id = ?", arrayOf(id.toString()))
            }
            for (id in lost) {
                dropCrops(db, "id = ?", arrayOf(id.toString()))
                db.delete("faces", "id = ?", arrayOf(id.toString()))
            }
            if (lost.isNotEmpty()) db.execSQL("DELETE FROM people WHERE id NOT IN (SELECT DISTINCT person FROM faces WHERE person IS NOT NULL)")
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        version++
    }

    /**
     * Before regrouping with a new recognizer: people the user never named or hid are let go
     * (their faces wait to be grouped again); named and hidden people, and every Not This Person,
     * stay as they are.
     */
    fun releaseUnnamed() {
        val db = writableDatabase
        db.beginTransaction()
        try {
            val unnamed = "SELECT id FROM people WHERE (name IS NULL OR name = '') AND hidden = 0"
            db.execSQL("UPDATE faces SET person = NULL WHERE person IN ($unnamed)")
            db.execSQL("DELETE FROM people WHERE id IN ($unnamed)")
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        version++
    }

    /**
     * Every person with at least one face, most photos first. [shown]: whether the app still shows
     * a photo or video. The cover comes only from those, so deleting a person's cover gives them
     * the next best one at once, instead of losing them until the face pass catches up.
     */
    fun people(shown: (mediaId: Long) -> Boolean = { true }): List<Person> {
        val names = HashMap<Long, Pair<String?, Boolean>>()
        val chosen = HashMap<Long, Long>()
        readableDatabase.rawQuery("SELECT id, name, hidden, cover_face FROM people", null).use { c ->
            while (c.moveToNext()) {
                names[c.getLong(0)] = (if (c.isNull(1)) null else c.getString(1)) to (c.getInt(2) != 0)
                if (!c.isNull(3)) chosen[c.getLong(0)] = c.getLong(3)
            }
        }
        class Acc(val media: HashSet<Long> = HashSet(), var faces: Int = 0) {
            var cover = 0L
            var face = 0L
            var video = false
            var box: List<Float> = emptyList()
            var score = -1f
        }
        val acc = HashMap<Long, Acc>()
        readableDatabase.rawQuery("SELECT person, media_id, l, t, r, b, id, at_ms FROM faces WHERE person IS NOT NULL", null).use { c ->
            while (c.moveToNext()) {
                val a = acc.getOrPut(c.getLong(0)) { Acc() }
                a.media += c.getLong(1)
                a.faces++
                if (!shown(c.getLong(1))) continue
                val box = listOf(c.getFloat(2), c.getFloat(3), c.getFloat(4), c.getFloat(5))
                val video = !c.isNull(7)
                // The cover the user chose; else the biggest face, from a photo when there is one
                // (sharper than a video frame).
                val score = if (chosen[c.getLong(0)] == c.getLong(6)) 10f
                else (box[2] - box[0]) * (box[3] - box[1]) + if (video) 0f else 1f
                if (score > a.score) {
                    a.score = score
                    a.cover = c.getLong(1)
                    a.face = c.getLong(6)
                    a.video = video
                    a.box = box
                }
            }
        }
        return acc.mapNotNull { (id, a) ->
            val (name, hidden) = names[id] ?: return@mapNotNull null
            Person(id, name, hidden, a.media, a.faces, a.cover, a.box, a.face, a.video)
        }.sortedByDescending { it.mediaIds.size }
    }

    /**
     * The people in one photo or video, biggest face first, each with the moment they first
     * show (videos; null for photos).
     */
    fun peopleIn(mediaId: Long): List<Pair<Long, Long?>> {
        val out = LinkedHashMap<Long, Long?>()
        readableDatabase.rawQuery(
            "SELECT person, at_ms FROM faces WHERE media_id = ? AND person IS NOT NULL ORDER BY (r - l) * (b - t) DESC",
            arrayOf(mediaId.toString()),
        ).use { c ->
            while (c.moveToNext()) {
                val person = c.getLong(0)
                val at = if (c.isNull(1)) null else c.getLong(1)
                val earliest = out[person]
                if (person !in out || (at != null && (earliest == null || at < earliest))) out[person] = at
            }
        }
        return out.toList()
    }

    /** Names a person; blank clears the name. */
    /** Every face in one photo or video, biggest first, whether or not it has a person yet. */
    fun facesIn(mediaId: Long): List<FaceIn> = ArrayList<FaceIn>().apply {
        readableDatabase.rawQuery(
            "SELECT id, person, l, t, r, b, at_ms FROM faces WHERE media_id = ? ORDER BY (r - l) * (b - t) DESC",
            arrayOf(mediaId.toString()),
        ).use { c ->
            while (c.moveToNext()) {
                add(
                    FaceIn(
                        c.getLong(0), if (c.isNull(1)) null else c.getLong(1),
                        listOf(c.getFloat(2), c.getFloat(3), c.getFloat(4), c.getFloat(5)), if (c.isNull(6)) null else c.getLong(6),
                    ),
                )
            }
        }
    }

    /**
     * A person of just [face] (one still waiting for a person), so the user can name someone the
     * grouping couldn't place: after it's named, grouping adds their other faces that match.
     */
    fun startPerson(face: FaceIn, mediaId: Long): Person {
        val db = writableDatabase
        val person = db.insert("people", null, ContentValues().apply { put("hidden", 0) })
        db.execSQL("UPDATE faces SET person = ? WHERE id = ? AND person IS NULL", arrayOf(person, face.id))
        version++
        return Person(person, null, false, setOf(mediaId), 1, mediaId, face.box, face.id, face.atMs != null)
    }

    /** Undoes [startPerson] when the user didn't name them after all: the face waits again. */
    fun unstartPerson(person: Long) {
        val db = writableDatabase
        db.execSQL("UPDATE faces SET person = NULL WHERE person = ?", arrayOf(person))
        db.delete("people", "id = ? AND name IS NULL", arrayOf(person.toString()))
        version++
    }

    fun rename(person: Long, name: String) {
        writableDatabase.execSQL("UPDATE people SET name = ? WHERE id = ?", arrayOf<Any?>(name.trim().ifEmpty { null }, person))
        version++
    }

    /**
     * Makes [gone] part of [keep]: its faces, and its name if [keep] has none. Faces told they
     * aren't [gone] aren't [keep] either.
     */
    /** Each person's name (null if unnamed) and whether they're hidden. */
    fun labels(): Map<Long, Pair<String?, Boolean>> = HashMap<Long, Pair<String?, Boolean>>().apply {
        readableDatabase.rawQuery("SELECT id, name, hidden FROM people", null).use { c ->
            while (c.moveToNext()) put(c.getLong(0), (if (c.isNull(1)) null else c.getString(1)) to (c.getInt(2) != 0))
        }
    }

    fun merge(keep: Long, gone: Long) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            db.execSQL("UPDATE faces SET person = ? WHERE person = ?", arrayOf(keep, gone))
            db.execSQL("UPDATE faces SET not_person = ? WHERE not_person = ?", arrayOf(keep, gone))
            db.execSQL(
                "UPDATE people SET name = (SELECT name FROM people WHERE id = ?) WHERE id = ? AND name IS NULL",
                arrayOf(gone, keep),
            )
            db.delete("people", "id = ?", arrayOf(gone.toString()))
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        version++
    }

    /** Not This Person: [person]'s faces in these photos leave them for good; an emptied person goes. */
    fun notThisPerson(person: Long, mediaIds: Collection<Long>) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            for (id in mediaIds) {
                db.execSQL(
                    "UPDATE faces SET person = NULL, not_person = ? WHERE person = ? AND media_id = ?",
                    arrayOf(person, person, id),
                )
            }
            db.execSQL("DELETE FROM people WHERE id = ? AND NOT EXISTS (SELECT 1 FROM faces WHERE person = ?)", arrayOf(person, person))
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        version++
    }

    /** Make Person Cover: [person]'s face in this photo or video (the biggest, if several). */
    fun setCover(person: Long, mediaId: Long) {
        writableDatabase.execSQL(
            "UPDATE people SET cover_face = (SELECT id FROM faces WHERE person = ? AND media_id = ? " +
                "ORDER BY (r - l) * (b - t) DESC LIMIT 1) WHERE id = ?",
            arrayOf(person, mediaId, person),
        )
        version++
    }

    fun setHidden(person: Long, hidden: Boolean) {
        writableDatabase.execSQL("UPDATE people SET hidden = ? WHERE id = ?", arrayOf(if (hidden) 1 else 0, person))
        version++
    }

    /** Saves grouping: faces joining existing people, and new people with their first faces. */
    fun assign(joins: Map<Long, Long>, newPeople: List<List<Long>>) {
        if (joins.isEmpty() && newPeople.isEmpty()) return
        val db = writableDatabase
        db.beginTransaction()
        try {
            for ((face, person) in joins) db.execSQL("UPDATE faces SET person = ? WHERE id = ?", arrayOf(person, face))
            for (faces in newPeople) {
                val person = db.insert("people", null, ContentValues().apply { put("hidden", 0) })
                for (face in faces) db.execSQL("UPDATE faces SET person = ? WHERE id = ?", arrayOf(person, face))
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        version++
    }

    companion object {
        /** [version] as a flow: People in Collections reloads when the face pass adds people. */
        val changes = kotlinx.coroutines.flow.MutableStateFlow(0)

        @Volatile
        private var open: FaceStore? = null

        /** One open helper for the whole process (it's thread-safe). */
        @Synchronized
        fun of(context: Context): FaceStore = open ?: FaceStore(context.applicationContext).also { open = it }

        @Synchronized
        fun delete(context: Context) {
            open?.close()
            open = null
            SQLiteDatabase.deleteDatabase(file(context))
            cropDir(context).deleteRecursively()
        }

        private fun cropDir(context: Context) = File(context.noBackupFilesDir, "faces").apply { mkdirs() }

        /** [crop] without opening the database, for showing a cover. */
        fun cropFile(context: Context, faceId: Long) = File(cropDir(context), "$faceId.jpg")

        private fun file(context: Context) =
            File(context.noBackupFilesDir, "search/faces.db").apply { parentFile?.mkdirs() }

        private fun toBytes(vector: FloatArray): ByteArray =
            ByteBuffer.allocate(vector.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(vector) }.array()
    }
}
