#!/bin/sh
# Publishes Lumen AI, signed, from this Mac. Secrets come from the macOS Keychain and GnuPG and
# only live in this process's environment.
#
#   scripts/publish.sh local     # to ~/.m2, to check the bundle
#   scripts/publish.sh central   # upload to the Central Portal; then press Publish there
#
# Keychain items: "Lumen AI signing key" (the key's passphrase) and, for central,
# "Maven Central token" (account: the token's username; password: the token's password).
set -e
cd "$(dirname "$0")/.."

KEY_ID=96A9D09D4E7E8071
export ORG_GRADLE_PROJECT_signingInMemoryKeyPassword="$(security find-generic-password -a lumen-ai -s 'Lumen AI signing key' -w)"
export ORG_GRADLE_PROJECT_signingInMemoryKey="$(gpg --batch --pinentry-mode loopback \
    --passphrase "$ORG_GRADLE_PROJECT_signingInMemoryKeyPassword" --export-secret-keys --armor "$KEY_ID")"

case "$1" in
  local)
    ./gradlew :ai:publishToMavenLocal --no-configuration-cache
    ;;
  central)
    export ORG_GRADLE_PROJECT_mavenCentralUsername="$(security find-generic-password -s 'Maven Central token' | sed -n 's/.*"acct"<blob>="\(.*\)"/\1/p')"
    export ORG_GRADLE_PROJECT_mavenCentralPassword="$(security find-generic-password -s 'Maven Central token' -w)"
    ./gradlew :ai:publishToMavenCentral --no-configuration-cache
    ;;
  *)
    echo "usage: $0 local|central" >&2
    exit 1
    ;;
esac
