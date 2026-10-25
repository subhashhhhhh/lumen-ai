package com.subhashh.lumen.ai.sample

import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.subhashh.lumen.ai.FeatureStatus
import com.subhashh.lumen.ai.IndexStatus
import com.subhashh.lumen.ai.LumenAi
import com.subhashh.lumen.ai.MediaRef
import com.subhashh.lumen.ai.MediaStoreLibrary
import com.subhashh.lumen.ai.ModelPack
import com.subhashh.lumen.ai.Person
import com.subhashh.lumen.ai.SearchResults
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * Lumen AI's sample: download features, watch indexing, search, see people, and tap a result for
 * its text and labels. Every smart feature here is one LumenAi call.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Box(Modifier.fillMaxSize().background(Color(0xFF111318)).safeDrawingPadding()) { Sample() }
            }
        }
    }
}

@Composable
private fun Sample() {
    val context = LocalContext.current
    var granted by remember { mutableStateOf(false) }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
        granted = result[Manifest.permission.READ_MEDIA_IMAGES] == true || result[Manifest.permission.READ_EXTERNAL_STORAGE] == true
        if (granted) LumenAi.indexing.start(context)
    }
    LaunchedEffect(Unit) {
        ask.launch(
            arrayOf(
                Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.READ_MEDIA_VIDEO,
                Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.POST_NOTIFICATIONS,
            ),
        )
    }
    if (!granted) {
        Text("Allow access to photos and videos to try Lumen AI.", Modifier.padding(24.dp), color = Color.White)
        return
    }

    // The library, by id, so a result can be opened (MediaStoreLibrary is the default library).
    val library by produceState(emptyMap<Long, MediaRef>()) { value = MediaStoreLibrary.load(context).associateBy { it.id } }
    var query by remember { mutableStateOf("") }
    val results by produceState(SearchResults.Empty, query) {
        delay(300)
        value = LumenAi.search(context, query)
    }
    val people by produceState(emptyList<Person>()) {
        while (true) {
            value = LumenAi.people.all(context)
            delay(5_000)
        }
    }
    var opened by remember { mutableStateOf<MediaRef?>(null) }

    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Lumen AI", style = MaterialTheme.typography.headlineMedium, color = Color.White)
        Features()
        Progress()
        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            placeholder = { Text("Search: cat on a sofa, \"exact words\", two names") },
        )
        if (query.isBlank() && people.isNotEmpty()) {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                items(people, key = { it.id }) { person -> Face(person) }
            }
        }
        if (query.isNotBlank()) {
            Text("${results.items.size} results", color = Color.Gray)
        }
        LazyVerticalGrid(GridCells.Adaptive(96.dp), horizontalArrangement = Arrangement.spacedBy(2.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            items(results.items, key = { it }) { id ->
                val item = library[id] ?: return@items
                Thumbnail(item, Modifier.aspectRatio(1f).clickable { opened = item })
            }
        }
    }
    opened?.let { item -> Details(item) { opened = null } }
}

/** One row per feature: its status, and a button to download it. */
@Composable
private fun Features() {
    val context = LocalContext.current
    // Status changes as downloads run: re-read it while one is running.
    val download by LumenAi.models.progress.collectAsStateWithLifecycle()
    var tick by remember { mutableStateOf(0) }
    LaunchedEffect(download) { while (true) { delay(1_000); tick++ } }
    for (feature in listOf(ModelPack.Search, ModelPack.Text, ModelPack.Faces)) {
        val status = remember(download, tick) { LumenAi.models.status(context, feature) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(feature.label, Modifier.width(120.dp), color = Color.White)
            Text(
                when (status) {
                    FeatureStatus.Ready -> "Ready"
                    FeatureStatus.Queued -> "Queued"
                    FeatureStatus.UpdateAvailable -> "Update available"
                    FeatureStatus.NotDownloaded -> "%,d MB".format(LumenAi.models.size(context, feature) / 1_000_000)
                    is FeatureStatus.Downloading -> "Downloading ${status.done * 100 / status.total.coerceAtLeast(1)}%"
                },
                Modifier.weight(1f),
                color = Color.Gray,
            )
            if (status == FeatureStatus.NotDownloaded || status == FeatureStatus.UpdateAvailable) {
                Button(onClick = { LumenAi.models.download(context, feature) }) { Text("Download") }
            }
        }
    }
}

@Composable
private fun Progress() {
    val status by LumenAi.indexing.progress.collectAsStateWithLifecycle(IndexStatus(null, null, null))
    val parts = listOfNotNull(
        status.search?.let { "Search ${it.indexed}/${it.total}" },
        status.text?.let { "Text ${it.indexed}/${it.total}" },
        status.faces?.let { "Faces ${it.indexed}/${it.total}" },
    )
    if (parts.isNotEmpty()) Text("Indexing: " + parts.joinToString(" · "), color = Color.Gray)
}

@Composable
private fun Face(person: Person) {
    val context = LocalContext.current
    val face by produceState<Bitmap?>(null, person.coverFaceId) {
        value = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(LumenAi.people.faceImage(context, person).path) }
    }
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(Modifier.size(64.dp).clip(CircleShape).background(Color.DarkGray)) {
            face?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
        }
        Text(person.name ?: "${person.mediaIds.size} photos", color = Color.White, style = MaterialTheme.typography.labelMedium)
    }
}

@Composable
private fun Thumbnail(item: MediaRef, modifier: Modifier) {
    val context = LocalContext.current
    val bitmap by produceState<Bitmap?>(null, item.id) {
        value = withContext(Dispatchers.IO) { runCatching { context.contentResolver.loadThumbnail(item.uri, Size(256, 256), null) }.getOrNull() }
    }
    Box(modifier.background(Color.DarkGray)) {
        bitmap?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
    }
}

/** A result's labels and text: LumenAi.labels and LumenAi.liveText. */
@Composable
private fun Details(item: MediaRef, onClose: () -> Unit) {
    val context = LocalContext.current
    val labels by produceState(emptyList<String>(), item.id) { value = LumenAi.labels(context, item.id) }
    val text by produceState<String?>(null, item.id) {
        value = LumenAi.liveText(context, item)?.lines?.joinToString("\n") { it.text } ?: ""
    }
    AlertDialog(
        onDismissRequest = onClose,
        confirmButton = { TextButton(onClick = onClose) { Text("Close") } },
        title = { Text(item.name) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Thumbnail(item, Modifier.fillMaxWidth().aspectRatio(1f))
                if (labels.isNotEmpty()) Text("Looks like: " + labels.joinToString(", "))
                Spacer(Modifier.size(4.dp))
                when (val t = text) {
                    null -> Text("Reading text…", color = Color.Gray)
                    "" -> Text("No text found.", color = Color.Gray)
                    else -> SelectionContainer { Text(t) }
                }
            }
        },
    )
}
