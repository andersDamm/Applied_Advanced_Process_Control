"""

OPC UA Database sql library
date: 07/09/2021
author: Anders Hilmar Damm Andersen

"""

import psycopg2


def connect():
	
	conn = None

	try:
		conn = psycopg2.connect(
			host     = "localhost",
			database = "OPC_UA_UTC",
			user     = "postgres",
			password = "Anders_SQL"
			)
		conn.autocommit = True
	except (Exception, psycopg2.DatabaseError) as error:
		print(error)

	return conn

def create_session(conn, u, r):
	cur = conn.cursor()

	s = "INSERT INTO runtime_session DEFAULT VALUES RETURNING rsid;"
	cur.execute(s)
	rsid = cur.fetchone()[0]
	q = """
		SELECT start_time
		FROM runtime_session
		WHERE rsid = %s
	"""
	cur.execute(q, [rsid,])
	timestamp = cur.fetchone()[0]

	print("RSID for this session = {}".format(rsid))
	print("Session created = {} (UTC)".format(timestamp))
	cur.close()
	inserty1(conn, rsid, timestamp,0.0,"Startup", 0)
	inserty2(conn, rsid, timestamp,0.0,"Startup",0)
	insertu1(conn, rsid, timestamp,u[0],"Startup", 0)
	insertu2(conn, rsid, timestamp,u[1],"Startup",0)
	insertr1(conn, rsid, timestamp,r[0], 0)
	insertr2(conn, rsid, timestamp,r[1], 0)
	insertControlmode(conn, rsid, timestamp, 0)
	return rsid

def inserty1(conn, rsid, timestamp, y1, statuscode, controlmode):
	
	cur = conn.cursor()
	row = [rsid, timestamp, y1, statuscode, controlmode]
	s = """
		INSERT INTO y1("rsid","time_stamp","y1","statuscode", "controlmode")
		VALUES(%s, %s, %s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()

def inserty2(conn, rsid, timestamp, y2, statuscode, controlmode):
	
	cur = conn.cursor()
	row = [rsid, timestamp, y2, statuscode, controlmode]

	s = """
		INSERT INTO y2("rsid","time_stamp","y2","statuscode", "controlmode")
		VALUES(%s, %s, %s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()


def insertu1(conn, rsid, timestamp, u1, statuscode, controlmode):
	
	cur = conn.cursor()
	row = [rsid, timestamp, u1, statuscode, controlmode]

	s = """
		INSERT INTO u1("rsid","time_stamp","u1","statuscode", "controlmode")
		VALUES(%s, %s, %s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()


def insertu2(conn, rsid, timestamp, u2, statuscode, controlmode):
	
	cur = conn.cursor()
	row = [rsid, timestamp, u2, statuscode, controlmode]

	s = """
		INSERT INTO u2("rsid","time_stamp","u2","statuscode", "controlmode")
		VALUES(%s, %s, %s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()

def insertr1(conn, rsid, timestamp, r1, controlmode):
	
	cur = conn.cursor()
	row = [rsid, timestamp, r1, controlmode]

	s = """
		INSERT INTO r1("rsid","time_stamp","r1", "controlmode")
		VALUES(%s, %s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()

def insertr2(conn, rsid, timestamp, r2, controlmode):
	
	cur = conn.cursor()
	row = [rsid, timestamp, r2, controlmode]

	s = """
		INSERT INTO r2("rsid","time_stamp","r2", "controlmode")
		VALUES(%s, %s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()

def insertControlmode(conn, rsid, timestamp, controlmode):
	
	cur = conn.cursor()
	row = [rsid, timestamp, controlmode]
	s = """
		INSERT INTO Controlmode("rsid","time_stamp", "controlmode")
		VALUES(%s, %s, %s);
	"""
	cur.execute(s, row)
	cur.close()

def gety1(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT y1, time_stamp, StatusCode, Controlmode
		FROM y1
		WHERE rsid = %s
		ORDER BY time_stamp DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	row = cur.fetchall()
	cur.close()
	return row[0]

def gety2(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT y2, time_stamp, StatusCode, Controlmode
		FROM y2
		WHERE rsid = %s
		ORDER BY time_stamp DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	row = cur.fetchall()
	cur.close()
	return row[0]

def getu1(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT u1, time_stamp, StatusCode, Controlmode
		FROM u1
		WHERE rsid = %s
		ORDER BY time_stamp DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	row = cur.fetchall()
	cur.close()
	return row[0]

def getu2(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT u2, time_stamp, StatusCode, Controlmode
		FROM u2
		WHERE rsid = %s
		ORDER BY time_stamp DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	row = cur.fetchall()
	cur.close()
	return row[0]

def getr1(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT r1, time_stamp, Controlmode
		FROM r1
		WHERE rsid = %s
		ORDER BY time_stamp DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	row = cur.fetchall()
	cur.close()
	return row[0]

def getr2(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT r2, time_stamp, Controlmode
		FROM r2
		WHERE rsid = %s
		ORDER BY time_stamp DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	row = cur.fetchall()
	cur.close()
	return row[0]

def getControlmode(conn, id):
	cur = conn.cursor()
	
	s = """
		SELECT Controlmode, time_stamp
		FROM Controlmode
		WHERE rsid = %s
		ORDER BY time_stamp DESC
		LIMIT 1
	"""
	cur.execute(s, [id,])
	row = cur.fetchall()
	cur.close()
	return row[0]


def getRSID(conn):
	# Fetch the most recent RSID
	cur = conn.cursor()
	s = """
		SELECT RSID
		FROM runtime_session
		ORDER BY start_time DESC
		LIMIT 1
	"""
	cur.execute(s)
	result = cur.fetchone()[0]
	cur.close()

	return result

if __name__ == "__main__":

	conn = connect()
	rsid = create_session(conn,[10, 10], [7, 14])
	rsid = getRSID(conn)
	print(gety1(conn, rsid)[0])
	print(gety2(conn, rsid)[1])
	print(gety2(conn, rsid)[2])
	print(gety2(conn, rsid)[3])
	print("Get control mode: ", getControlmode(conn, rsid)[0])




