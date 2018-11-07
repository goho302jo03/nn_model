const express = require("express")
const exec = require('child_process').exec
const app = express()
const port = 49999

app.listen(port)
app.use(express.static(__dirname + "/public"))

app.get("/generate", function(req, res) {
  var hair = req.query.hair
  var eye = req.query.eye
  exec('python3 public/demo_web.py ' + hair + ' ' + eye, function(error, stdout, stderr){
    console.log(error)
    console.log(stderr)
    console.log(stdout)
  })
  res.send(`${hair} ${eye}`)
})
