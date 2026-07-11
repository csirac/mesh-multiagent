
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-load firebase_admin to avoid import errors if not installed
_firebase_app = None


def _init_firebase(credentials_file: str | Path) -> bool:
    """
    Initialize Firebase Admin SDK.

    Args:
        credentials_file: Path to service account JSON file

    Returns:
        True if initialization succeeded, False otherwise
    """
    global _firebase_app

    if _firebase_app is not None:
        return True

    try:
        import firebase_admin
        from firebase_admin import credentials

        from .paths import resolve_path
        cred_path = Path(resolve_path(str(credentials_file)))
        if not cred_path.exists():
            logger.error(f"FCM credentials file not found: {cred_path}")
            return False

        cred = credentials.Certificate(str(cred_path))
        _firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized")
        return True

    except ImportError:
        logger.error("firebase-admin package not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return False


class FCMSender:
    """
    Sends push notifications via Firebase Cloud Messaging.

    Usage:
        sender = FCMSender("/path/to/credentials.json")
        if sender.is_available:
            await sender.send_notification(fcm_token, title, body, data)
    """

    def __init__(self, credentials_file: str | Path | None = None):
        """
        Initialize FCM sender.

        Args:
            credentials_file: Path to Firebase service account JSON.
                             If None, FCM will be disabled.
        """
        self._enabled = False
        self._credentials_file = credentials_file

        if credentials_file:
            self._enabled = _init_firebase(credentials_file)

    @property
    def is_available(self) -> bool:
        """Check if FCM is properly configured and available."""
        return self._enabled

    async def send_notification(
        self,
        fcm_token: str,
        title: str,
        body: str,
        data: dict[str, str] | None = None,
    ) -> bool:
        """
        Send a push notification to a device.

        Args:
            fcm_token: The device's FCM registration token
            title: Notification title
            body: Notification body text
            data: Optional data payload (all values must be strings)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._enabled:
            logger.warning("FCM not enabled, skipping notification")
            return False

        try:
            from firebase_admin import messaging

            # Build the message
            notification = messaging.Notification(
                title=title,
                body=body,
            )

            # Android-specific config for high priority
            android_config = messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="mesh_messages",
                ),
            )

            message = messaging.Message(
                notification=notification,
                data=data or {},
                token=fcm_token,
                android=android_config,
            )

            # Send synchronously (firebase_admin doesn't have async API)
            # In production, this should be done in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                messaging.send,
                message,
            )

            logger.info(f"FCM notification sent: {response}")
            return True

        except Exception as e:
            logger.error(f"Failed to send FCM notification: {e}")
            return False

    async def send_data_message(
        self,
        fcm_token: str,
        data: dict[str, str],
    ) -> bool:
        """
        Send a data-only message (no visible notification).

        This is useful for silent updates that the app handles internally.

        Args:
            fcm_token: The device's FCM registration token
            data: Data payload (all values must be strings)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._enabled:
            logger.warning("FCM not enabled, skipping data message")
            return False

        try:
            from firebase_admin import messaging

            android_config = messaging.AndroidConfig(
                priority="high",
            )

            message = messaging.Message(
                data=data,
                token=fcm_token,
                android=android_config,
            )

            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                messaging.send,
                message,
            )

            logger.info(f"FCM data message sent: {response}")
            return True

        except Exception as e:
            logger.error(f"Failed to send FCM data message: {e}")
            return False
