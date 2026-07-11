# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.

# Keep kotlinx.serialization classes
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt

-keepclassmembers class kotlinx.serialization.json.** {
    *** Companion;
}
-keepclasseswithmembers class kotlinx.serialization.json.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# Keep protocol classes
-keep,includedescriptorclasses class com.mesh.client.data.remote.protocol.**$$serializer { *; }
-keepclassmembers class com.mesh.client.data.remote.protocol.** {
    *** Companion;
}
-keepclasseswithmembers class com.mesh.client.data.remote.protocol.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# Keep Room entities
-keep class com.mesh.client.data.local.db.entities.** { *; }

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-dontnote okhttp3.**
-dontnote okio.**
