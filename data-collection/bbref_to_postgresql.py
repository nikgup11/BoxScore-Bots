import psycopg2

conn = psycopg2.connect(
    database="bsb_pd",
    user="elipappas",
    password="postgres",
    host="localhost",  # e.g., "localhost" or an IP address
    port="5432"    # e.g., "5432"
)

cursor = conn.cursor()