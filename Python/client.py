"""

client for FTS_server to read/write to database.

"""


from opcua import Client
import time
from tzlocal import get_localzone
import pytz
import OPCUA_SQL as db

if __name__ == "__main__":

	Ts = 0.1
	url = 'opc.tcp://localhost:4840/FTS'
	client = Client(url)
	client.connect()
	objects = client.get_objects_node()

	y1 = objects.get_children()[1].get_children()[0]
	y2 = objects.get_children()[2].get_children()[0]

	u1 = objects.get_children()[3].get_children()[0]
	u2 = objects.get_children()[4].get_children()[0]
	
	# Connect to database with most recent RSID
	conn = db.connect()
	rsid = db.getRSID(conn)
	print("Client RSID = {}".format(rsid))
	print("Start OPC UA Client")
	try:
		while True:
			# ---- Fetch data from y1 and y2 with timestamp and statuscode ----
			cm = db.getControlmode(conn, rsid)[0]
			db.inserty1(conn, rsid, objects.get_children()[1].get_children()[0].get_data_value().SourceTimestamp\
				, y1.get_value(),str(objects.get_children()[1].get_children()[0].get_data_value().StatusCode), cm)
			db.inserty2(conn, rsid, objects.get_children()[2].get_children()[0].get_data_value().SourceTimestamp\
				, y2.get_value(),str(objects.get_children()[2].get_children()[0].get_data_value().StatusCode), cm)

			# ---- Read u1 and u2 from database and write them to server ----

			u1.set_value(db.getu1(conn, rsid)[0])
			u2.set_value(db.getu2(conn, rsid)[0])

			# ---- Sleep for Ts seconds ----
			time.sleep(Ts-(time.time()%Ts))
	finally:
		print("Stopping Client")
		client.close_session()
		print("Client stopped")