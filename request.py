import requests

url = 'http://localhost:5000/predict'
r = requests.post(url,json={'comment':"Hey wassup"})

print(r.json())