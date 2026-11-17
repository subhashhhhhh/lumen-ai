#!/bin/sh
# Builds the speech library (src/main/cpp) for both ARM ABIs and packs it for machines without an
# NDK, such as a Linux ARM build server (see lumenAi.nativeLibs in ai/build.gradle.kts).
#
#   scripts/package-native.sh [output folder]     default: build/native-prebuilt
#
# Needs the NDK, so run it on a Mac or an x86 Linux machine. SKIP_BUILD=1 packs what the last
# release build left in ai/build instead of building again. Run it again whenever src/main/cpp
# changes: builds that use the output check its SOURCE_HASH against the sources.
set -eu
cd "$(dirname "$0")/.."
out="${1:-build/native-prebuilt}"
built=ai/build/intermediates/library_jni/release/copyReleaseJniLibsProjectOnly/jni

[ "${SKIP_BUILD:-}" = 1 ] || ./gradlew -q :ai:copyReleaseJniLibsProjectOnly
[ -f "$built/arm64-v8a/liblumen_speech.so" ] || { echo "No built library in $built: run without SKIP_BUILD." >&2; exit 1; }

strip=$(ls "${ANDROID_HOME:-$HOME/Library/Android/sdk}"/ndk/*/toolchains/llvm/prebuilt/*/bin/llvm-strip 2>/dev/null | tail -1)
rm -rf "$out"
for abi in arm64-v8a armeabi-v7a; do
    mkdir -p "$out/$abi"
    cp "$built/$abi/liblumen_speech.so" "$out/$abi/"
    # As the app's own packaging would strip it, where there is an NDK to do it with.
    [ -z "$strip" ] || "$strip" --strip-unneeded "$out/$abi/liblumen_speech.so"
done
# The hash ai/build.gradle.kts computes: of each source file's hash line, in sorted order.
(cd ai/src/main/cpp && find . -type f | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -d" " -f1) > "$out/SOURCE_HASH"
echo "Packed into $out (sources $(cat "$out/SOURCE_HASH" | cut -c1-12)):"
ls -l "$out"/*/
