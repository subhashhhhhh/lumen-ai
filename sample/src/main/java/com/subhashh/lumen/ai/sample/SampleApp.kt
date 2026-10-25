package com.subhashh.lumen.ai.sample

import android.app.Application
import com.subhashh.lumen.ai.LumenAi
import com.subhashh.lumen.ai.SearchModel

class SampleApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // The whole setup. Everything else in this sample only calls LumenAi.
        LumenAi.install(this) {
            searchModel = SearchModel.Fast   // 240 MB instead of 670 MB
        }
        LumenAi.indexing.start(this)
    }
}
