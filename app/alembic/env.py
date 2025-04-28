from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# --- Load .env variables ---
import os
from dotenv import load_dotenv

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Alembic config ---
config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL.replace('%', '%%'))

# --- Logging config ---
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- Import and gather metadata ---
import sys
from importlib import import_module

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

model_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
for filename in os.listdir(model_folder):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = f'models.{filename[:-3]}'
        import_module(module_name)

from models.chat import Base
target_metadata = Base.metadata

# --- Migrations ---
def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url, target_metadata=target_metadata,
        literal_binds=True, dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
