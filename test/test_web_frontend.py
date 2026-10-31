'''
The web frontend of the running simulations (iteras.scripts.web_frontend),
started by the iteras_frontend_start and specula_frontend_start commands.
'''
import specula
specula.init(-1)

import os
import unittest
from unittest.mock import patch

from iteras.scripts import web_frontend


def _env_without_port():
    return {k: v for k, v in os.environ.items() if k not in ('ITERAS_PORT', 'SPECULA_PORT')}


class TestWebFrontend(unittest.TestCase):

    def test_port(self):
        with patch.dict(os.environ, _env_without_port(), clear=True):
            self.assertEqual(web_frontend.port(), 8080)
            os.environ['SPECULA_PORT'] = '8081'
            self.assertEqual(web_frontend.port(), 8081)
            os.environ['ITERAS_PORT'] = '8082'
            self.assertEqual(web_frontend.port(), 8082)

    def test_index_page(self):
        response = web_frontend.app.test_client().get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Running simulations', response.data)

    def test_simulation_updates_are_forwarded(self):
        '''An update sent by a simulation is forwarded to the browsers'''
        client = web_frontend.socketio.test_client(web_frontend.app)
        try:
            client.emit('simul_update', {'name': 'sim', 'status': 'running'})
            received = client.get_received()
        finally:
            client.disconnect()
        self.assertIn(('simul_update', [{'name': 'sim', 'status': 'running'}]),
                      [(r['name'], r['args']) for r in received])

    def test_start(self):
        with patch.dict(os.environ, {'ITERAS_PORT': '8090'}), \
                patch.object(web_frontend, 'daemonize') as daemonize, \
                patch.object(web_frontend.socketio, 'run') as run:
            web_frontend.start()
        if hasattr(os, 'fork'):
            daemonize.assert_called_once()
        self.assertEqual(run.call_args.kwargs['port'], 8090)

    @unittest.skipUnless(hasattr(os, 'fork'), 'stop() is not supported on this platform')
    def test_stop_stops_both_commands(self):
        with patch.object(web_frontend, 'killProcessByName') as kill:
            web_frontend.stop()
        self.assertEqual([c.args[0] for c in kill.call_args_list],
                         ['iteras_frontend_start', 'specula_frontend_start'])


if __name__ == '__main__':
    unittest.main()
