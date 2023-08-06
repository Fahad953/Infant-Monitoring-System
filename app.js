// const db=require("./database");
const express = require('express');
const routes=require('./routes.js');
const bodyParser = require('body-parser');
const app = express();
// app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.urlencoded({ extended: false }));
const port = 3000; // Choose a port number you want your server to listen on
//routes
app.use(routes);

app.use(bodyParser.json());
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  next();
});
app.use(express.json());
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
