const Router=require("express");

const db=require("../database.js")

const router=Router();


router.get('/register/:name/:email/:pass', (req,res)=>{
            console.log(req.params, "params");
            const q = `call project.Register_User('${req.params.name}','${req.params.email}','${req.params.pass}')`;
            console.log(q,"query");
            db.query(q,(err,result)=>{
                        if(err){
                            console.log(err,"error");
                        }
                        else{
                            console.log(result[0],"result");
                            res.send(result[0]);
                        }
            })
          
});

router.get('/login/:email/:pass', (req,res)=>{
    console.log(req.params, "params");
    const q = `call project.login_User('${req.params.email}','${req.params.pass}')`;
    console.log(q,"query");
    db.query(q,(err,result)=>{
                if(err){
                    console.log(err,"error");
                }
                else{
                    console.log(result[0],"result");
                    res.send(result[0]);
                }
    })
   
  
})


module.exports=router;