import fs from 'fs'
import csv from 'csv-parser'

var csvData = {
    loadData: (fileName) => {
        fs.createReadStream(fileName)
        .pipe(csv())
        .on('data', (data) => {
            
        })
    }
}

module.exports = csvData