"""
Plaid API client for banking data access.

This client provides access to bank accounts, transactions, and balances via the Plaid API.
It handles:
- Link token generation for user authentication
- Access token management (secure storage)
- Transaction syncing with local caching
- Multi-institution support

Configuration:
    ~/.config/mesh/plaid.yaml:
        plaid:
            client_id: "your-client-id"
            secret: "your-secret"
            environment: "sandbox"  # or "development", "production"
            redirect_uri: "https://your-server.com/plaid/callback"  # Optional
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

from ..paths import resolve_path

DEFAULT_CONFIG_PATH = resolve_path("~/.config/mesh/plaid.yaml")
DEFAULT_TOKENS_DB = resolve_path("~/.config/mesh/plaid_tokens.db")
DEFAULT_CACHE_DB = resolve_path("~/log/plaid/transactions.db")


def _load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load Plaid configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path) as f:
        config = yaml.safe_load(f)

    return config.get("plaid", {})


# =============================================================================
# Token Storage
# =============================================================================

class PlaidTokenStore:
    """Secure storage for Plaid access tokens using SQLite."""

    def __init__(self, db_path: str = DEFAULT_TOKENS_DB):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_tokens (
                    user_id TEXT NOT NULL,
                    institution_id TEXT NOT NULL,
                    institution_name TEXT,
                    access_token TEXT NOT NULL,
                    item_id TEXT,
                    cursor TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, institution_id)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def store_token(
        self,
        user_id: str,
        institution_id: str,
        institution_name: str,
        access_token: str,
        item_id: str | None = None,
    ) -> None:
        """Store or update an access token."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO access_tokens (user_id, institution_id, institution_name, access_token, item_id, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, institution_id) DO UPDATE SET
                    access_token = excluded.access_token,
                    item_id = excluded.item_id,
                    institution_name = excluded.institution_name,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, institution_id, institution_name, access_token, item_id))
            conn.commit()
        finally:
            conn.close()

    def get_token(self, user_id: str, institution_id: str) -> dict[str, Any] | None:
        """Get an access token by user and institution."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("""
                SELECT * FROM access_tokens
                WHERE user_id = ? AND institution_id = ?
            """, (user_id, institution_id)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_institutions(self, user_id: str) -> list[dict[str, Any]]:
        """List all linked institutions for a user."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("""
                SELECT institution_id, institution_name, created_at, updated_at
                FROM access_tokens
                WHERE user_id = ?
                ORDER BY institution_name
            """, (user_id,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete_token(self, user_id: str, institution_id: str) -> bool:
        """Delete an access token."""
        conn = sqlite3.connect(self.db_path)
        try:
            result = conn.execute("""
                DELETE FROM access_tokens
                WHERE user_id = ? AND institution_id = ?
            """, (user_id, institution_id))
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def update_cursor(self, user_id: str, institution_id: str, cursor: str) -> None:
        """Update the sync cursor for an institution."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE access_tokens
                SET cursor = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND institution_id = ?
            """, (cursor, user_id, institution_id))
            conn.commit()
        finally:
            conn.close()


# =============================================================================
# Transaction Cache
# =============================================================================

class PlaidTransactionCache:
    """Local cache for Plaid transactions using SQLite."""

    def __init__(self, db_path: str = DEFAULT_CACHE_DB):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            # Accounts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    institution_id TEXT NOT NULL,
                    name TEXT,
                    official_name TEXT,
                    type TEXT,
                    subtype TEXT,
                    mask TEXT,
                    current_balance REAL,
                    available_balance REAL,
                    currency TEXT DEFAULT 'USD',
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Transactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    datetime TEXT,
                    name TEXT,
                    merchant_name TEXT,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    category TEXT,
                    category_id TEXT,
                    pending INTEGER DEFAULT 0,
                    payment_channel TEXT,
                    location_city TEXT,
                    location_state TEXT,
                    raw_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                )
            """)

            # Indexes for fast queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_account ON transactions(account_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_user ON transactions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_user_date ON transactions(user_id, date)")

            conn.commit()
        finally:
            conn.close()

    def upsert_account(self, user_id: str, institution_id: str, account: dict) -> None:
        """Insert or update an account."""
        conn = sqlite3.connect(self.db_path)
        try:
            balances = account.get("balances", {})
            conn.execute("""
                INSERT INTO accounts (
                    account_id, user_id, institution_id, name, official_name,
                    type, subtype, mask, current_balance, available_balance, currency, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(account_id) DO UPDATE SET
                    name = excluded.name,
                    official_name = excluded.official_name,
                    type = excluded.type,
                    subtype = excluded.subtype,
                    mask = excluded.mask,
                    current_balance = excluded.current_balance,
                    available_balance = excluded.available_balance,
                    currency = excluded.currency,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                account["account_id"],
                user_id,
                institution_id,
                account.get("name"),
                account.get("official_name"),
                account.get("type"),
                account.get("subtype"),
                account.get("mask"),
                balances.get("current"),
                balances.get("available"),
                balances.get("iso_currency_code", "USD"),
            ))
            conn.commit()
        finally:
            conn.close()

    def upsert_transaction(self, user_id: str, txn: dict) -> None:
        """Insert or update a transaction."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Handle category as a list
            category = txn.get("category")
            if isinstance(category, list):
                category = " > ".join(category)

            # Handle location
            location = txn.get("location", {}) or {}

            conn.execute("""
                INSERT INTO transactions (
                    transaction_id, account_id, user_id, date, datetime, name, merchant_name,
                    amount, currency, category, category_id, pending, payment_channel,
                    location_city, location_state, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO UPDATE SET
                    name = excluded.name,
                    merchant_name = excluded.merchant_name,
                    amount = excluded.amount,
                    category = excluded.category,
                    pending = excluded.pending,
                    raw_json = excluded.raw_json
            """, (
                txn["transaction_id"],
                txn["account_id"],
                user_id,
                txn["date"],
                txn.get("datetime"),
                txn.get("name"),
                txn.get("merchant_name"),
                txn["amount"],
                txn.get("iso_currency_code", "USD"),
                category,
                txn.get("category_id"),
                1 if txn.get("pending") else 0,
                txn.get("payment_channel"),
                location.get("city"),
                location.get("region"),
                json.dumps(txn),
            ))
            conn.commit()
        finally:
            conn.close()

    def delete_transaction(self, transaction_id: str) -> None:
        """Delete a transaction (for handling removed transactions from Plaid)."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM transactions WHERE transaction_id = ?", (transaction_id,))
            conn.commit()
        finally:
            conn.close()

    def get_transactions(
        self,
        user_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        account_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query cached transactions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = "SELECT * FROM transactions WHERE user_id = ?"
            params: list = [user_id]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            if account_id:
                query += " AND account_id = ?"
                params.append(account_id)

            query += " ORDER BY date DESC, datetime DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_accounts(self, user_id: str) -> list[dict]:
        """Get all accounts for a user."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("""
                SELECT * FROM accounts WHERE user_id = ?
                ORDER BY institution_id, name
            """, (user_id,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


# =============================================================================
# Plaid API Client
# =============================================================================

class PlaidClient:
    """
    Client for Plaid API interactions.

    Provides methods for:
    - Link token generation (for user auth flow)
    - Public token exchange
    - Account listing
    - Transaction sync
    - Balance queries
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        tokens_db: str = DEFAULT_TOKENS_DB,
        cache_db: str = DEFAULT_CACHE_DB,
        user_id: str = "default",
    ):
        self.config = _load_config(config_path)
        self.user_id = user_id
        self.token_store = PlaidTokenStore(tokens_db)
        self.cache = PlaidTransactionCache(cache_db)

        self._plaid_client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Plaid API client."""
        if not self.config:
            logger.warning("Plaid configuration not found")
            return

        try:
            import plaid
            from plaid.api import plaid_api

            client_id = self.config.get("client_id")
            secret = self.config.get("secret")
            environment = self.config.get("environment", "sandbox")

            if not client_id or not secret:
                logger.warning("Plaid client_id or secret not configured")
                return

            # Map environment to Plaid host
            # Note: plaid-python v38+ only has Sandbox and Production
            if environment == "production":
                host = plaid.Environment.Production
            else:
                # sandbox and development both use Sandbox in newer SDK
                host = plaid.Environment.Sandbox

            configuration = plaid.Configuration(
                host=host,
                api_key={
                    "clientId": client_id,
                    "secret": secret,
                }
            )

            api_client = plaid.ApiClient(configuration)
            self._plaid_client = plaid_api.PlaidApi(api_client)
            logger.info(f"Plaid client initialized ({environment})")

        except ImportError:
            logger.warning("plaid-python package not installed. Run: pip install plaid-python")
        except Exception as e:
            logger.error(f"Failed to initialize Plaid client: {e}")

    def is_available(self) -> bool:
        """Check if the Plaid client is initialized and ready."""
        return self._plaid_client is not None

    # -------------------------------------------------------------------------
    # Link Flow
    # -------------------------------------------------------------------------

    def get_link_token(
        self,
        products: list[str] | None = None,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a Link token for the user authentication flow.

        Args:
            products: Plaid products to request (default: ["transactions"])
            redirect_uri: OAuth redirect URI (required for OAuth institutions)

        Returns:
            Dict with link_token, expiration, and link_url
        """
        if not self.is_available():
            return {"error": "Plaid client not initialized"}

        try:
            from plaid.model.link_token_create_request import LinkTokenCreateRequest
            from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
            from plaid.model.products import Products
            from plaid.model.country_code import CountryCode

            if products is None:
                products = ["transactions"]

            # Build request
            request = LinkTokenCreateRequest(
                user=LinkTokenCreateRequestUser(client_user_id=self.user_id),
                client_name="Mesh Finance",
                products=[Products(p) for p in products],
                country_codes=[CountryCode("US")],
                language="en",
            )

            # Add redirect URI if provided
            if redirect_uri or self.config.get("redirect_uri"):
                request.redirect_uri = redirect_uri or self.config.get("redirect_uri")

            response = self._plaid_client.link_token_create(request)

            return {
                "link_token": response.link_token,
                "expiration": str(response.expiration),
                "request_id": response.request_id,
            }

        except Exception as e:
            logger.error(f"Failed to create link token: {e}")
            return {"error": str(e)}

    def exchange_public_token(
        self,
        public_token: str,
        institution_id: str,
        institution_name: str,
    ) -> dict[str, Any]:
        """
        Exchange a public token (from Link) for an access token.

        Args:
            public_token: The public token from Plaid Link
            institution_id: Institution ID (e.g., "ins_123")
            institution_name: Human-readable institution name

        Returns:
            Dict with access_token and item_id, or error
        """
        if not self.is_available():
            return {"error": "Plaid client not initialized"}

        try:
            from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest

            request = ItemPublicTokenExchangeRequest(public_token=public_token)
            response = self._plaid_client.item_public_token_exchange(request)

            access_token = response.access_token
            item_id = response.item_id

            # Store the token
            self.token_store.store_token(
                user_id=self.user_id,
                institution_id=institution_id,
                institution_name=institution_name,
                access_token=access_token,
                item_id=item_id,
            )

            return {
                "success": True,
                "item_id": item_id,
                "institution_id": institution_id,
                "institution_name": institution_name,
            }

        except Exception as e:
            logger.error(f"Failed to exchange public token: {e}")
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # Accounts
    # -------------------------------------------------------------------------

    def list_linked_institutions(self) -> list[dict]:
        """List all institutions the user has linked."""
        return self.token_store.list_institutions(self.user_id)

    def get_accounts(self, institution_id: str | None = None) -> list[dict]:
        """
        Get accounts for linked institutions.

        If institution_id is provided, only get accounts for that institution.
        Otherwise, get accounts for all linked institutions.
        """
        if not self.is_available():
            return [{"error": "Plaid client not initialized"}]

        try:
            from plaid.model.accounts_get_request import AccountsGetRequest

            institutions = self.list_linked_institutions()
            if institution_id:
                institutions = [i for i in institutions if i["institution_id"] == institution_id]

            all_accounts = []

            for inst in institutions:
                token_info = self.token_store.get_token(self.user_id, inst["institution_id"])
                if not token_info:
                    continue

                request = AccountsGetRequest(access_token=token_info["access_token"])
                response = self._plaid_client.accounts_get(request)

                for account in response.accounts:
                    account_dict = account.to_dict()
                    account_dict["institution_id"] = inst["institution_id"]
                    account_dict["institution_name"] = inst["institution_name"]
                    all_accounts.append(account_dict)

                    # Update cache
                    self.cache.upsert_account(self.user_id, inst["institution_id"], account_dict)

            return all_accounts

        except Exception as e:
            logger.error(f"Failed to get accounts: {e}")
            return [{"error": str(e)}]

    def get_balances(self, institution_id: str | None = None) -> list[dict]:
        """Get current balances for all accounts."""
        accounts = self.get_accounts(institution_id)

        results = []
        for acc in accounts:
            if "error" in acc:
                results.append(acc)
                continue

            balances = acc.get("balances", {})
            results.append({
                "account_id": acc.get("account_id"),
                "name": acc.get("name"),
                "type": acc.get("type"),
                "subtype": acc.get("subtype"),
                "mask": acc.get("mask"),
                "institution_name": acc.get("institution_name"),
                "current": balances.get("current"),
                "available": balances.get("available"),
                "currency": balances.get("iso_currency_code", "USD"),
            })

        return results

    # -------------------------------------------------------------------------
    # Transactions
    # -------------------------------------------------------------------------

    def sync_transactions(self, institution_id: str | None = None) -> dict[str, Any]:
        """
        Sync transactions from Plaid using the Transactions Sync API.

        This uses cursors to efficiently sync only new/modified transactions.
        Synced transactions are stored in the local cache.

        Returns:
            Dict with counts of added, modified, removed transactions
        """
        if not self.is_available():
            return {"error": "Plaid client not initialized"}

        try:
            from plaid.model.transactions_sync_request import TransactionsSyncRequest

            institutions = self.list_linked_institutions()
            if institution_id:
                institutions = [i for i in institutions if i["institution_id"] == institution_id]

            total_added = 0
            total_modified = 0
            total_removed = 0

            for inst in institutions:
                token_info = self.token_store.get_token(self.user_id, inst["institution_id"])
                if not token_info:
                    continue

                access_token = token_info["access_token"]
                cursor = token_info.get("cursor")

                has_more = True
                while has_more:
                    request = TransactionsSyncRequest(
                        access_token=access_token,
                        cursor=cursor,
                    )
                    response = self._plaid_client.transactions_sync(request)

                    # Process added transactions
                    for txn in response.added:
                        txn_dict = txn.to_dict()
                        self.cache.upsert_transaction(self.user_id, txn_dict)
                        total_added += 1

                    # Process modified transactions
                    for txn in response.modified:
                        txn_dict = txn.to_dict()
                        self.cache.upsert_transaction(self.user_id, txn_dict)
                        total_modified += 1

                    # Process removed transactions
                    for txn in response.removed:
                        self.cache.delete_transaction(txn.transaction_id)
                        total_removed += 1

                    # Update accounts
                    for account in response.accounts:
                        account_dict = account.to_dict()
                        self.cache.upsert_account(self.user_id, inst["institution_id"], account_dict)

                    cursor = response.next_cursor
                    has_more = response.has_more

                # Save cursor for next sync
                self.token_store.update_cursor(self.user_id, inst["institution_id"], cursor)

            return {
                "added": total_added,
                "modified": total_modified,
                "removed": total_removed,
                "institutions_synced": len(institutions),
            }

        except Exception as e:
            logger.error(f"Failed to sync transactions: {e}")
            return {"error": str(e)}

    def get_transactions(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        account_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query transactions from the local cache.

        Args:
            start_date: Start date (YYYY-MM-DD), defaults to 30 days ago
            end_date: End date (YYYY-MM-DD), defaults to today
            account_id: Filter to specific account
            limit: Max transactions to return

        Returns:
            List of transaction dicts
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        return self.cache.get_transactions(
            user_id=self.user_id,
            start_date=start_date,
            end_date=end_date,
            account_id=account_id,
            limit=limit,
        )

    # -------------------------------------------------------------------------
    # Unlink
    # -------------------------------------------------------------------------

    def unlink_institution(self, institution_id: str) -> dict[str, Any]:
        """
        Unlink an institution (revoke access token).

        This removes the stored token but keeps cached transactions.
        """
        if not self.is_available():
            return {"error": "Plaid client not initialized"}

        try:
            from plaid.model.item_remove_request import ItemRemoveRequest

            token_info = self.token_store.get_token(self.user_id, institution_id)
            if not token_info:
                return {"error": f"No linked institution found: {institution_id}"}

            # Revoke access at Plaid
            request = ItemRemoveRequest(access_token=token_info["access_token"])
            self._plaid_client.item_remove(request)

            # Delete local token
            self.token_store.delete_token(self.user_id, institution_id)

            return {
                "success": True,
                "institution_id": institution_id,
                "message": "Institution unlinked successfully",
            }

        except Exception as e:
            logger.error(f"Failed to unlink institution: {e}")
            return {"error": str(e)}
