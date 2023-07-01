# This is a placeholder for database_interface.py
import sqlite3 

class DatabaseInterface:
    """
    This class provides methods to access and manipulate data in a SQLite database file. 
    """
 
    def __init__(self, db_file):
        """
        Initializes a database interface object with a specified database file.
        """
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()

    def close_connection(self):
        """
        Closes the connection to the database.
        """
        self.conn.close()

    def execute_query(self, query):
        """
        Executes an SQL query on the current database connection.
        """
        self.cursor.execute(query)
        self.conn.commit()
        rows = self.cursor.fetchall()
        return rows
 
    def execute_non_query(self, non_query):
        """
        Executes an SQL query on the current database connection that does not return any results.
        """
        self.cursor.execute(non_query)
        self.conn.commit()

    def execute_many(self, many_query):
        """
        Executes a query with multiple sets of values.
        """
        self.cursor.executemany(many_query)
        self.conn.commit()
