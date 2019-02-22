import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs';
import csv from 'csvtojson'
import express from 'express'

const app = express()
const model = tf.sequential()

app.listen(3000)
app.get('/predict/:x1/:x2/:x3/:x4', (req, res) => {
    console.log(req.params.x1, req.params.x2, req.params.x3, req.params.x4);
    let predict = model.predict(tf.tensor2d([[parseFloat(req.params.x1), parseFloat(req.params.x2), parseFloat(req.params.x3), parseFloat(req.params.x4)]]));
    let output = {
        'Iris-setosa': (predict.get(0, 0) * 100),
        'Iris-versicolor': (predict.get(0, 1) * 100),
        'Iris-virginica': (predict.get(0, 2) * 100)
    }
    
    res.json(output);
})


csv().fromFile('iris-train.csv')
.then((jsonObj)=>{
    train(jsonObj)
})


//training
function train(data) {
    const trainingData = tf.tensor2d(data.map(item => [parseFloat(item.SepalLengthCm), parseFloat(item.SepalWidthCm), parseFloat(item.PetalLengthCm), parseFloat(item.PetalWidthCm)]))
    const labels = tf.tensor2d(data.map(data => [
        data.Species == 'Iris-setosa' ? 1 : 0,
        data.Species == 'Iris-versicolor' ? 1 : 0,
        data.Species == 'Iris-virginica' ? 1 : 0,
    ]))
    
    network(trainingData, labels)
}

//testing
function test(data) {
    const trainingData = tf.tensor2d(data.map(item => [parseFloat(item.SepalLengthCm), parseFloat(item.SepalWidthCm), parseFloat(item.PetalLengthCm), parseFloat(item.PetalWidthCm)]))
    
    return trainingData
}

function network(trainingData, labels) {
    tf.tidy(() => {
        model.add(tf.layers.dense({
            inputShape: [4],
            activation: "sigmoid",
            units: 4
        }))
        
        model.add(tf.layers.dense({
            inputShape: [4],
            activation: "sigmoid",
            units: 4
        }))

        model.add(tf.layers.dense({
            inputShape: [4],
            activation: "sigmoid",
            units: 3
        }))
        
        
        model.add(tf.layers.dense({
            activation: "tanh",
            units: 3
        }))
        
        model.compile({
            loss: "meanSquaredError",
            optimizer: tf.train.adam(.06)
        })
        
        model.fit(trainingData, labels, {epochs: 500}).then((history) => {
            csv().fromFile('iris-test.csv')
            .then((jsonObj)=>{
                model.predict(test(jsonObj)).print()
            })
        })
    })
    
}

