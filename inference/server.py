from transformers import YolosForObjectDetection
import torch
import os

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import io


DEVICE_STRING = "mps"

os.environ['YOLO_MODE'] = 'SERVER'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
server_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device=DEVICE_STRING)


class YOLORequestHandler(BaseHTTPRequestHandler):
    def _set_response(self, content_length=0):
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        if content_length != 0:
            self.send_header('Content-Length', f"{content_length}")
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        buff = io.BytesIO()
        buff.write(post_data)
        self._set_response()
        buff.seek(0)
        input_object = torch.load(buff, map_location=DEVICE_STRING)
        embedd = input_object.type(torch.float32)
        result = server_model(None, input_embeddings=embedd, p_height=512, p_width=682)

        buffer = io.BytesIO()
        torch.save(result, buffer)
        buffer.seek(0)
        self.wfile.write(buffer.read())


def run(server_class=HTTPServer, handler_class=YOLORequestHandler, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()