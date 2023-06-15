import requests

headers = {
    'Content-type': 'text/plain',
}

response = requests.post('http://localhost:8080/invocations', headers=headers)
