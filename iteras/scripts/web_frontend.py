
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from iteras import registry
from iteras.lib.process_utils import daemonize, killProcessByName


app = Flask('Iteras_frontend', template_folder = os.path.join(os.path.dirname(__file__), 'templates'))
socketio = SocketIO(app)

# Commands that start the frontend (see pyproject.toml), stopped by stop()
START_COMMANDS = ['iteras_frontend_start', 'specula_frontend_start']

'''
From clients:

import socketio
sio_client = socketio.Client()
sio_client.connect('http://localhost:8080')
sio_client.emit('msg', {"key": "value"})
'''

@socketio.on('simul_update')
def simul_update(data):
    '''
    Message received from running simulations with: (name, data, port)
    '''
    client_id = request.sid
    print(client_id, data)
    socketio.emit('simul_update', data)

@app.route('/')
def index():
    return render_template('frontend.html')

def port():
    '''Port of the frontend: ITERAS_PORT (or one of its aliases, like SPECULA_PORT), or 8080'''
    return int(registry.getenv('ITERAS_PORT', '8080'))

def start():
    print(f'Starting the simulation frontend server on port {port()}')
    if hasattr(os, 'fork'):
        daemonize()
    else:
        print('Warning: cannot run in background on this platform, running in foreground. Press Ctrl-C to stop.')
    socketio.run(app, host='0.0.0.0', port=port(), allow_unsafe_werkzeug=True)

def stop():
    if not hasattr(os, 'fork'):
        print('Not supported on this platform: stop the frontend with Ctrl-C in its terminal.')
        return
    for command in START_COMMANDS:
        killProcessByName(command)
   
if __name__ == '__main__':
    start()
