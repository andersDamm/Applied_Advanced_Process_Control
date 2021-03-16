"""
Python script to interact with the Database AAPC_test1

date: 06/03/2021
author: Anders Hilmar Damm Andersen
"""

import psycopg2


def connect():
	
	conn = None

	try:
		conn = psycopg2.connect(
			host     = "localhost",
			database = "AAPC_test2",
			user     = "postgres",
			password = "Anders_SQL"
			)
		conn.autocommit = True
	except (Exception, psycopg2.DatabaseError) as error:
		print(error)

	return conn

def create_session(conn):
	cur = conn.cursor()

	s = "INSERT INTO runtime_session DEFAULT VALUES RETURNING rsid;"
	cur.execute(s)
	id = cur.fetchone()[0]
	
	print("Id",id)
	
	cur.close()
	insert_meas(conn, id, 0.0, 0.0, 0.0) # Init measurements
	insert_input(conn, id, 0.0, 0.0, 0.0, 0.0) # Init measurements
	return id

def get_u(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT u
		FROM Data
		WHERE rsid = %s
		ORDER BY time DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	u = cur.fetchone()[0]
	cur.close()

	return u
def get_time_from_meas(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT Time
		FROM Meas
		WHERE rsid = %s
		ORDER BY time DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	time = cur.fetchone()[0]
	cur.close()

	return time
def get_time_from_input(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT Time
		FROM Data
		WHERE rsid = %s
		ORDER BY time DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	time = cur.fetchone()[0]
	cur.close()

	return time

def get_states(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT x1, x2
		FROM Meas
		WHERE rsid = %s
		ORDER BY time DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	row = cur.fetchone()
	cur.close()

	return (row[0], row[1])

def insert_meas(conn, id, time, x1, x2):
	
	cur = conn.cursor()
	row = [id, time, x1, x2]

	s = """
		INSERT INTO Meas("rsid","time","x1","x2")
		VALUES(%s, %s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()

def insert_input(conn, id, time, x1, x2, u):
	
	cur = conn.cursor()
	row = [id, time, x1, x2, u]

	s = """
		INSERT INTO Data("rsid","time","x1","x2","u")
		VALUES(%s, %s, %s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()

def read_meas(conn, id, nrow):
	cur = conn.cursor()
	
	s = """
		SELECT time, x1, x2
		FROM Meas
		WHERE rsid = %s
		ORDER BY time DESC
		LIMIT %s
	"""
	cur.execute(s, [id, nrow])
	result = cur.fetchall()
	cur.close()

	result.reverse()

	return result

def readAll_meas(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT time, x1, x2
		FROM Meas
		WHERE rsid = %s
		ORDER BY time DESC
	"""
	cur.execute(s, [id])
	result = cur.fetchall()
	cur.close()

	result.reverse()

	return result
def read_input(conn, id, nrow):
	cur = conn.cursor()
	
	s = """
		SELECT time, u
		FROM Data
		WHERE rsid = %s
		ORDER BY time DESC
		LIMIT %s
	"""
	cur.execute(s, [id, nrow])
	result = cur.fetchall()
	cur.close()

	result.reverse()

	return result

def readAll_input(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT time, u
		FROM Data
		WHERE rsid = %s
		ORDER BY time DESC
	"""
	cur.execute(s, [id])
	result = cur.fetchall()
	cur.close()

	result.reverse()

	return result

def readAllFromView_meas(conn):
	cur = conn.cursor()
	
	s = """
		SELECT time, x1, x2
		FROM resent_session_meas
		ORDER BY time DESC
	"""
	cur.execute(s)
	result = cur.fetchall()
	cur.close()

	result.reverse()

	return result

def readAllFromView_input(conn):
	cur = conn.cursor()
	
	s = """
		SELECT time, u
		FROM resent_session_meas_input
		ORDER BY time DESC
	"""
	cur.execute(s)
	result = cur.fetchall()
	cur.close()

	result.reverse()

	return result
def main():

	conn = connect()
	
	id = create_session(conn)
	u=get_u(conn, id)
	print(u)

	(x1, x2) = get_states(conn, id)

	insert_meas(conn, id, 0.012, 666.213, 666.123)
	insert_input(conn, id, 0.012, 1,1, u)
	data = read_meas(conn, id, 1)
	print(data)
if __name__ == '__main__':
	main()