"""
Database setup and management CLI.

Usage:
    # Initialize database (create tables)
    python -m server.db.setup init
    
    # Sync existing pipeline outputs to DB
    python -m server.db.setup sync
    
    # Create admin user
    python -m server.db.setup create-admin
    
    # Create a sales user
    python -m server.db.setup create-user --username john --name "John Doe" --email john@lvmh.com --role sales
    
    # Reset database (DESTRUCTIVE)
    python -m server.db.setup reset
    
    # Check status
    python -m server.db.setup status
"""
import argparse
import sys
import getpass
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_init():
    """Initialize database schema."""
    from server.db.schema import init_database
    init_database()
    print("✅ Database schema initialized")


def cmd_reset():
    """Reset database (drop and recreate all tables)."""
    confirm = input("⚠️  This will DELETE ALL DATA. Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        return
    from server.db.schema import reset_database
    reset_database()
    print("✅ Database reset complete")


def cmd_sync():
    """Sync existing pipeline file outputs to the database."""
    from server.db.sync import sync_results_to_db
    result = sync_results_to_db()
    print(f"✅ Sync complete: {result}")


def cmd_status():
    """Check database connection and show stats."""
    from server.db.connection import get_sync_connection, sync_cursor
    try:
        with sync_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM clients WHERE is_deleted = FALSE")
            clients = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM segments")
            segments = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM users")
            users = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM upload_sessions")
            uploads = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM pipeline_runs")
            runs = cur.fetchone()[0]

        print("✅ Database connection: OK")
        print(f"   Clients:   {clients}")
        print(f"   Segments:  {segments}")
        print(f"   Users:     {users}")
        print(f"   Uploads:   {uploads}")
        print(f"   Runs:      {runs}")

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)


def cmd_create_admin():
    """Create admin user interactively."""
    import bcrypt

    username = input("Admin username [admin]: ").strip() or "admin"
    display_name = input("Display name [Administrator]: ").strip() or "Administrator"
    email = input("Email [admin@lvmh.local]: ").strip() or "admin@lvmh.local"
    password = getpass.getpass("Password: ")
    if not password:
        print("Password cannot be empty")
        return

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    from server.db.connection import sync_cursor
    with sync_cursor() as cur:
        cur.execute("""
            INSERT INTO users (username, display_name, email, password_hash, role)
            VALUES (%s, %s, %s, %s, 'admin')
            ON CONFLICT (username) DO UPDATE SET
                password_hash = EXCLUDED.password_hash,
                display_name = EXCLUDED.display_name,
                email = EXCLUDED.email
            RETURNING id
        """, (username, display_name, email, password_hash))
        user_id = cur.fetchone()[0]

    print(f"✅ Admin user '{username}' created (id={user_id})")


def cmd_create_user(username, name, email, role, password=None):
    """Create a user."""
    import bcrypt

    if not password:
        password = getpass.getpass(f"Password for {username}: ")
    if not password:
        print("Password cannot be empty")
        return

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    from server.db.connection import sync_cursor
    with sync_cursor() as cur:
        cur.execute("""
            INSERT INTO users (username, display_name, email, password_hash, role)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (username) DO UPDATE SET
                password_hash = EXCLUDED.password_hash,
                display_name = EXCLUDED.display_name,
                email = EXCLUDED.email,
                role = EXCLUDED.role
            RETURNING id
        """, (username, name, email, password_hash, role))
        user_id = cur.fetchone()[0]

    print(f"✅ User '{username}' ({role}) created (id={user_id})")


def main():
    parser = argparse.ArgumentParser(description="LVMH Database Management")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("init", help="Initialize database schema")
    sub.add_parser("reset", help="Reset database (DESTRUCTIVE)")
    sub.add_parser("sync", help="Sync pipeline outputs to DB")
    sub.add_parser("status", help="Check DB connection and stats")
    sub.add_parser("create-admin", help="Create admin user")

    user_p = sub.add_parser("create-user", help="Create a new user")
    user_p.add_argument("--username", required=True)
    user_p.add_argument("--name", required=True, help="Display name")
    user_p.add_argument("--email", default="")
    user_p.add_argument("--role", default="sales", choices=["admin", "sales", "manager", "viewer"])
    user_p.add_argument("--password", default=None, help="Password (prompted if not provided)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "init":
        cmd_init()
    elif args.command == "reset":
        cmd_reset()
    elif args.command == "sync":
        cmd_sync()
    elif args.command == "status":
        cmd_status()
    elif args.command == "create-admin":
        cmd_create_admin()
    elif args.command == "create-user":
        cmd_create_user(args.username, args.name, args.email, args.role, args.password)


if __name__ == "__main__":
    main()
