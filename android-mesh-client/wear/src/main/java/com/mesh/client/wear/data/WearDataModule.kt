package com.mesh.client.wear.data

import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

/**
 * Hilt module for wear data layer dependencies.
 * WearMessageRepository is already @Singleton, so no additional bindings needed.
 */
@Module
@InstallIn(SingletonComponent::class)
object WearDataModule
