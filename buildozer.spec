[app]
title = LiveTTS Reader
package.name = livetts
package.domain = org.livetts
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,wav,json,onnx,pt,bin,token
version = 0.1.0
requirements = python3,kivy==2.3.0,numpy,ebooklib,beautifulsoup4,lxml,pyjnius
orientation = portrait
fullscreen = 1
android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE,READ_MEDIA_AUDIO
android.api = 33
android.minapi = 24
android.ndk = 27c
android.archs = arm64-v8a
android.gradle_dependencies = org.pytorch:pytorch_android_lite:2.0.0,org.pytorch:pytorch_android_torchvision_lite:2.0.0
p4a.branch = develop
p4a.hostpython3 = python3
android.allow_backup = True
ios.kivy_ios_branch = master
