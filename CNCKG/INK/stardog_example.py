import stardog
from pathlib import Path

# specify our endpoint, username, and password
# default values are provided, be sure to change if necessary
conn_details = {
    'endpoint': 'http://localhost:5820',
    'username': 'admin',
    'password': 'admin'
}

# create a new admin connection
with stardog.Admin(**conn_details) as admin:
    # create a new database
    db = admin.new_database('pythondb')
    print('Created db')

    db.drop()
    print('Dropped db')