const Router = require('express');


const AuthRoute=require("./routes/Auth.js");
const InfantRoute=require("./routes/infant.js");
const PythonModel=require("./routes/model.js");



const router = Router();


router.use("/api/auth", AuthRoute);
router.use("/api/infant", InfantRoute);
router.use("/api/model", PythonModel);

module.exports = router;