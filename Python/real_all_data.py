# Read all data from Database

import SQL_Python as sql

import matplotlib.pyplot as plt

from ver3 import data2collumns


def main(id = None):

    try:
        plt.close('all')
        mainConn = sql.connect()
        results = sql.readAllFromView(mainConn)
        for row in results:
            print(row)
        data = data2collumns(results, 4)
        print(data[3])
        print(len(data[0]))

        plt.clf()


        plt.subplot(211)
        plt.title('RSID: {}'.format(000), fontsize=20)
        axes = plt.gca()
        plt.plot(data[0], data[1],c = 'b', label='y')
        plt.legend(loc='upper left')
        plt.xlabel('time [s]', fontsize=14)
        plt.grid()

        plt.subplot(212)
        axes = plt.gca()
        plt.step(data[0], data[3],c = 'r', label='u')  # Change such that type is correctly updated.
        plt.legend(loc='upper left')
        plt.xlabel('time [s]', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.show()
    except (Exception) as error:
            print(error)
    finally:
            mainConn.close()

if __name__ == '__main__':
    main(99)

