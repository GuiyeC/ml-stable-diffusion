#!/bin/bash

pyinstaller GuernikaModelConverter/GuernikaModelConverter.spec --clean -y
codesign -s 'F3CA2B3DA0C3FA37A8F800B0EDCEF3923C36E753' -v --deep --force --options=runtime --timestamp --entitlements GuernikaModelConverter/GuernikaModelConverter.entitlements -o runtime "dist/Guernika Model Converter.app"
ditto "dist/Guernika Model Converter.app" "dist/pkg/Applications/Guernika Model Converter.app"
productbuild --identifier "com.guiyec.GuernikaModelConverter" --sign "01F6504F86D22EB924A51B2DFA28E5483A3172B6" --timestamp --root dist/pkg / dist/GuernikaModelConverter.pkg
rm -rf dist/pkg
#xcrun notarytool store-credentials --apple-id "guiyec@gmail.com" --team-id "A5ZC2LG374"
xcrun notarytool submit dist/GuernikaModelConverter.pkg --keychain-profile "com.guiyec.notarytool" --wait
xcrun stapler staple dist/GuernikaModelConverter.pkg
