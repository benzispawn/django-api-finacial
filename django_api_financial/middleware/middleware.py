from django.db import connections
import psycopg2.pool

from django_api_financial.settings import DB_USER, DB_HOST, DB_PORT, DB_PASSWORD, DB_NAME, DB_PATH_CERT


class PostgreSQLPoolMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=20,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            sslmode='require',
            sslrootcert=DB_PATH_CERT,
        )

    def __call__(self, request):
        # Get a connection from the pool
        connection = self.pool.getconn()
        connections['default'].connection = connection

        response = self.get_response(request)

        # Return connection to the pool
        self.pool.putconn(connection)

        return response
