# Module lumen-ai

On-device smart photo features for Android apps: search by description, text in photos and
videos, people, labels and subtitles. Nothing leaves the phone.

Start with [com.subhashh.lumen.ai.LumenAi]: install it in `Application.onCreate`, download the
features you want with `LumenAi.models`, and call `LumenAi.search`, `LumenAi.liveText`,
`LumenAi.people` and the rest.

Classes marked [com.subhashh.lumen.ai.LumenAiEngine] are the lower-level engine. They need an
opt-in, and their API can change in any release.

# Package com.subhashh.lumen.ai

Everything in Lumen AI: the simple API ([com.subhashh.lumen.ai.LumenAi]) and the engine behind it.
