package com.mesh.client.di

import android.content.Context
import com.mesh.client.data.local.db.MessageDao
import com.mesh.client.data.local.db.MeshDatabase
import com.mesh.client.data.remote.MeshSocketClient
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideMeshDatabase(@ApplicationContext context: Context): MeshDatabase {
        return MeshDatabase.getInstance(context)
    }

    @Provides
    @Singleton
    fun provideMeshSocketClient(): MeshSocketClient {
        return MeshSocketClient()
    }

    @Provides
    @Singleton
    fun provideMessageDao(database: MeshDatabase): MessageDao {
        return database.messageDao()
    }
}
