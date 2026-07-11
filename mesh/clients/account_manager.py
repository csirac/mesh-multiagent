
import json
import os
from dataclasses import dataclass
from typing import Optional, Any, Dict
from .gmail_client import GmailClient
from .calendar_client import CalendarClient

@dataclass
class AccountConfig:
    gmail_credentials: str
    gmail_token: str

    # Settings (global, not per-account)
    log_dir: Optional[str] = None
    google_api_key: Optional[str] = None
    synthetic_api_key: Optional[str] = None
    zai_api_key: Optional[str] = None
    exa_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    server_port: Optional[int] = None
    editor: Optional[str] = None
    web_storage_dir: Optional[str] = None
    user: Optional[str] = None

    def get_gmail_credentials(self) -> str:
        """Get Gmail credentials path, allowing environment variable override."""
        return os.getenv("GMAIL_CREDENTIALS", self.gmail_credentials)

    def get_gmail_token(self) -> str:
        """Get Gmail token path, allowing environment variable override."""
        return os.getenv("GMAIL_TOKEN", self.gmail_token)

    @staticmethod
    def _get_with_fallback(config_value: Optional[str], env_var: str) -> Optional[str]:
        """Get config value, falling back to environment variable."""
        if config_value is not None:
            return config_value
        return os.getenv(env_var)

    def get_log_dir(self) -> Optional[str]:
        return self._get_with_fallback(self.log_dir, "GPT_CHAT_LOG_DIR")

    def get_google_api_key(self) -> Optional[str]:
        return self._get_with_fallback(self.google_api_key, "GOOGLE_API_KEY")

    def get_synthetic_api_key(self) -> Optional[str]:
        return self._get_with_fallback(self.synthetic_api_key, "SYNTHETIC_API_KEY")

    def get_zai_api_key(self) -> Optional[str]:
        return self._get_with_fallback(self.zai_api_key, "ZAI_API_KEY")

    def get_exa_api_key(self) -> Optional[str]:
        return self._get_with_fallback(self.exa_api_key, "EXA_API_KEY")

    def get_openai_api_key(self) -> Optional[str]:
        return self._get_with_fallback(self.openai_api_key, "OPENAI_API_KEY")

    def get_server_port(self) -> Optional[int]:
        if self.server_port is not None:
            return self.server_port
        port = os.getenv("CHAT_SERVER_PORT")
        return int(port) if port else None

    def get_editor(self) -> Optional[str]:
        return self._get_with_fallback(self.editor, "EDITOR")

    def get_web_storage_dir(self) -> Optional[str]:
        return self._get_with_fallback(self.web_storage_dir, "WEB_STORAGE_DIR")

    def get_user(self) -> Optional[str]:
        return self._get_with_fallback(self.user, "USER")

class AccountManager:
    def __init__(self, config_path: str):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            self.default_account: str = raw.get("default_account", "work")

            # Extract global settings (applies to all accounts)
            settings = raw.get("settings", {})

            # Create account configs, merging global settings with per-account settings
            self.accounts: dict[str, AccountConfig] = {}
            for name, account_cfg in raw.get("accounts", {}).items():
                # Merge: global settings + account-specific settings
                merged_cfg = {**settings, **account_cfg}
                self.accounts[name] = AccountConfig(**merged_cfg)

            # Store global settings for direct access (returns settings from default account)
            self._settings = settings

        except Exception as e:
            print("Error: could not load configuration file for email/notes accounts.")
            print(e)

            self.default_account = None
            self.accounts = None
            self._settings = {}


    def list_accounts( self ):
        return list( self.accounts.keys() )

    def get(self, account: Optional[str]) -> AccountConfig:
        if self.accounts == None:
            return None

        key = account or self.default_account
        if key not in self.accounts:
            raise KeyError(f"Unknown account '{key}'. Known: {list(self.accounts.keys())}")
        return self.accounts[key]

    # Convenience methods for accessing global settings from default account
    def get_log_dir(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_log_dir() if cfg else None

    def get_google_api_key(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_google_api_key() if cfg else None

    def get_synthetic_api_key(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_synthetic_api_key() if cfg else None

    def get_zai_api_key(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_zai_api_key() if cfg else None

    def get_exa_api_key(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_exa_api_key() if cfg else None

    def get_openai_api_key(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_openai_api_key() if cfg else None

    def get_server_port(self) -> Optional[int]:
        cfg = self.get(self.default_account)
        return cfg.get_server_port() if cfg else None

    def get_editor(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_editor() if cfg else None

    def get_web_storage_dir(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_web_storage_dir() if cfg else None

    def get_user(self) -> Optional[str]:
        cfg = self.get(self.default_account)
        return cfg.get_user() if cfg else None

class ToolHost:
    def __init__(self, accounts_config_path: str, confirmation_mode: str = "cli"):
        self.confirmation_mode = confirmation_mode
        self.accounts = AccountManager(accounts_config_path)
        self.current_account: str = self.accounts.default_account

        # caches
        self._gmail_by_account = {}
        self._calendar_by_account = {}
        self._notes_by_account = {}
        self._papers_by_account = {}

        # set initial clients
        self._ensure_account_loaded(self.current_account)

    def _ensure_account_loaded(self, account: str) -> None:
        cfg = self.accounts.get(account)
        if cfg is None:
            self.accounts = None
            return None

        if account not in self._gmail_by_account:
            self._gmail_by_account[account] = GmailClient(
                credentials_file=cfg.get_gmail_credentials(),
                token_file=cfg.get_gmail_token(),
                confirmation_mode=self.confirmation_mode,
            )
            self._calendar_by_account[account] = CalendarClient(
                self._gmail_by_account[account].creds,
                confirmation_mode=self.confirmation_mode,
            )

    def set_current_account(self, account: str) -> None:
        # validate + lazy init
        _ = self.accounts.get(account)
        self._ensure_account_loaded(account)
        self.current_account = account

    def list_accounts(self):
        return self.accounts.list_accounts()

    def get_current_account(self):
        return self.current_account
        
    # Properties your existing tools can keep using (minimal changes)
    def gmail_client(self):
        return self._gmail_by_account[self.current_account]

    def calendar_client(self):
        return self._calendar_by_account[self.current_account]

    def notes_client(self):
        return self._notes_by_account.get(self.current_account)

    def papers_client(self):
        return self._papers_by_account.get(self.current_account)

    def clients(self):
        pc = self.papers_client()
        gc = self.gmail_client()
        nc = self.notes_client()
        cc = self.calendar_client()
        return (gc,cc,nc,pc)

    def close(self):
        """
        Best-effort cleanup for all per-account resources.
        Safe to call multiple times.
        """

        # Close Research Notes DBs for all accounts
        for acct, notes in list(getattr(self, "_notes_by_account", {}).items()):
            db = getattr(notes, "db", None)
            if db is not None and hasattr(db, "close"):
                try:
                    db.close()
                except Exception:
                    pass

        # Close Papers DBs for all accounts
        for acct, papers in list(getattr(self, "_papers_by_account", {}).items()):
            db = getattr(papers, "db", None)
            if db is not None and hasattr(db, "close"):
                try:
                    db.close()
                except Exception:
                    pass

        # If your wrappers themselves have close() methods, close them too (optional)
        for acct, notes in list(getattr(self, "_notes_by_account", {}).items()):
            if hasattr(notes, "close"):
                try:
                    notes.close()
                except Exception:
                    pass

        for acct, papers in list(getattr(self, "_papers_by_account", {}).items()):
            if hasattr(papers, "close"):
                try:
                    papers.close()
                except Exception:
                    pass

        # Gmail/Calendar clients typically don't hold open resources, but if yours do:
        for acct, g in list(getattr(self, "_gmail_by_account", {}).items()):
            if hasattr(g, "close"):
                try:
                    g.close()
                except Exception:
                    pass

        for acct, c in list(getattr(self, "_calendar_by_account", {}).items()):
            if hasattr(c, "close"):
                try:
                    c.close()
                except Exception:
                    pass

        # Clear caches
        self._notes_by_account = {}
        self._papers_by_account = {}
        self._gmail_by_account = {}
        self._calendar_by_account = {}


    def _parse_tool_args(self, raw: Any) -> Dict[str, Any]:
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}

    def _need_str(self, args: Dict[str, Any], key: str) -> str:
        v = args.get(key)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Missing/invalid '{key}' (expected non-empty string).")
        return v.strip()

    def _tool_msg(self, *, name: str, output: Any) -> dict:
        """
        Normalize tool output into your internal tool message format.
        `content` is always a string.
        """
        if isinstance(output, str):
            content = output
        else:
            content = json.dumps(output, ensure_ascii=False)
        return {"role": "tool", "name": name, "content": content}

    # ----------------------------
    # Dispatcher (account_* only)
    # ----------------------------
    def dispatch_tool_call(self, tool_call: dict) -> Optional[dict]:
        """
        tool_call shape:
          {"id": func_id, "function": {"name": name, "arguments": raw_json_string}}

        Returns:
          tool_msg dict if handled, else None.
        """
        fn = (tool_call or {}).get("function") or {}
        name = fn.get("name") or ""
        raw = fn.get("arguments")

        if not name.startswith("account_"):
            return None

        args = self._parse_tool_args(raw)

        try:
            if name == "account_list":
                out = {
                    "accounts": self.list_accounts(),
                    "current_account": self.get_current_account(),
                }
                return self._tool_msg(name=name, output=out)

            if name == "account_get_current":
                return self._tool_msg(
                    name=name,
                    output={"current_account": self.get_current_account()},
                )

            if name == "account_set_current":
                acct = self._need_str(args, "account")
                self.set_current_account(acct)
                return self._tool_msg(
                    name=name,
                    output={"ok": True, "current_account": self.get_current_account()},
                )

            return self._tool_msg(name=name, output={"error": f"Unknown account tool: {name}"})

        except Exception as e:
            # Keep tool name in message; encode error in content
            return self._tool_msg(name=name, output={"error": str(e)})
