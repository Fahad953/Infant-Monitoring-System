const Router = require('express');


const AuthRoute=require("./routes/Auth.js");



// module.exports=Router().use('/api/auth', AuthRoute);

module.exports = Router().use('/api/auth', AuthRoute)    