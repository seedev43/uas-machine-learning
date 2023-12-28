from flask import Flask, render_template, request
from datetime import datetime
import pickle
import numpy as np

# load file pickle nya
classificationPickle = './water_potability.pkl'
clusteringPickle = './bank_transaction.pkl'
with open(classificationPickle, 'rb') as file:
    classModel = pickle.load(file)
    print(f"File {classificationPickle} loaded!")

with open(clusteringPickle, 'rb') as file:
    clusterModel = pickle.load(file)
    print(f"File {clusteringPickle} loaded!")

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    datas = {
        # Mendapatkan alamat IP pengguna
        'user_ip' : request.headers.get('X-Forwarded-For', request.remote_addr),
        'ip': request.remote_addr,
        'time': datetime.now().strftime('%H:%M:%S')
    }
    return render_template('index.html', data=datas)

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    if request.method == "GET":
        return render_template('classification.html')
    # k = request.form['k_params']
    # if k != "":
    #     classModel.n_neighbors = int(k)
    # mengambil semua value dari form input html
    values = [float(x) for x in request.form.values()]
    # masukkan value tadi ke array
    array_values = [values]
    # print(array_values[0])
    prediction = classModel.predict(array_values)
    datas = {
        'inputs': array_values,
        'k_params': int(classModel.n_neighbors),
        'predict' : prediction[0]
    }
    return render_template('classification.html', data=datas)

@app.route("/clustering", methods=['GET', 'POST'])
def clustering():
    if request.method == "GET":
        return render_template('clustering.html')
    # k = request.form['k_params']
    # if k != "":
    #     clusterModel.n_clusters = int(k)
    # mengambil semua value dari form input html
    values = [float(x) for x in request.form.values()]
    # masukkan value tadi ke array
    array_values = [np.array(values)]
    prediction = clusterModel.predict(array_values)
    datas = {
        'inputs': array_values,
        'n_cluster': int(clusterModel.n_clusters),
        'cluster' : prediction,
        'centroids' : enumerate(clusterModel.cluster_centers_),
        'inertia': clusterModel.inertia_
    }
    return render_template('clustering.html', data=datas)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run(debug=True) 
    

