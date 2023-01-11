#!/bin/bash

pyinstaller GuernikaModelConverter/GuernikaModelConverter.spec --clean -y
codesign -s 'F3CA2B3DA0C3FA37A8F800B0EDCEF3923C36E753' -v --deep --force --options=runtime --timestamp --entitlements GuernikaModelConverter/GuernikaModelConverter.entitlements -o runtime "dist/Guernika Model Converter.app"
ditto "dist/Guernika Model Converter.app" "dist/dmg/Guernika Model Converter.app"
hdiutil create dist/tmp_guernikaconverter.dmg -ov -volname "GuernikaModelConverter" -fs APFS -srcfolder "dist/dmg/"
hdiutil convert dist/tmp_guernikaconverter.dmg -format UDZO -o dist/GuernikaModelConverter.dmg
codesign -s 'F3CA2B3DA0C3FA37A8F800B0EDCEF3923C36E753' --timestamp dist/GuernikaModelConverter.dmg
rm -rf dist/dmg
rm dist/tmp_guernikaconverter.dmg
#xcrun notarytool store-credentials --apple-id "guiyec@gmail.com" --team-id "A5ZC2LG374"
xcrun notarytool submit dist/GuernikaModelConverter.dmg --keychain-profile "com.guiyec.notarytool" --wait
xcrun stapler staple dist/GuernikaModelConverter.dmg
