import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs';
import csv from 'csvtojson'
import express from 'express'

const app = express()
const model = tf.sequential()

app.listen(3000)
app.get('/predict/:x1', (req, res) => {
    let predict = model.predict(tf.tensor2d([[parseInt(req.params.x1)]]));

    res.json({predict: Math.round(predict.get(0, 0))});
})


csv().fromFile('numbers-train.csv')
.then((jsonObj)=>{
    train(jsonObj)
})


//training
function train(data) {
    const trainingData = tf.tensor2d(data.map(item => [parseInt(item.valor)]))
    const labels = tf.tensor2d(data.map(data => [parseInt(data.label)]))
    
    network(trainingData, labels)
}

//testing
function test(data) {
    const trainingData = tf.tensor2d(data.map(item => [parseInt(item.valor)]))
    
    return trainingData
}

function network(trainingData, labels) {
    tf.tidy(() => {
        model.add(tf.layers.dense({
            inputShape: [1],
            activation: "linear",
            units: 5
        }))
        
        model.add(tf.layers.dense({
            inputShape: [5],
            activation: "linear",
            units: 5
        }))

        model.add(tf.layers.dense({
            inputShape: [5],
            activation: "linear",
            units: 5
        }))
        
        
        model.add(tf.layers.dense({
            activation: "linear",
            units: 1
        }))
        
        model.compile({
            loss: "meanSquaredError",
            optimizer: tf.train.adam(.01)
        })
        
        model.fit(trainingData, labels, {epochs: 400}).then((history) => {
            csv().fromFile('numbers-test.csv')
            .then((jsonObj)=>{
                model.predict(test(jsonObj)).print()
            })
        })
    })
    
}

