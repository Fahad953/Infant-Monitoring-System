const mysql = require('mysql');

const connection=mysql.createConnection({
    host:"localhost",
    user:"root",
    password:"123456",
    database:"project",
});

connection.connect((error)=>{
        if(error){
            console.error('Error connecting to the database:', error);
        }
        else{
            console.log('Connected to the database');
        }
});


module.exports = connection;