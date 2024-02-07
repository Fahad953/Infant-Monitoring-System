const express = require("express");
const db = require("../database");
const router = express.Router();

router.post("/addInfant", (req, res) => {
  console.log(req.body, "body");
  const { name, age,weight,height,sleep_hours,feeding,immune_status, bloodGroup, gender } = req.body.info;
  const { user_id } = req.body;
  const query = `call project.add_infant('${name}',${parseInt(age)},${parseFloat(weight)},${parseInt(height)},${parseInt(sleep_hours)},'${feeding}','${immune_status}','${bloodGroup}','${gender}',${user_id})`;
  console.log(query);
  db.query(query, (err, result) => {
    if (err) {
      console.log(err, "error");
    } else {
      console.log(result[0], "result");
      res.send(result[0]);
    }
  });
});

router.get("/display/:user_id", (req, res) => {
  console.log(req.params, "user_id");
  const { user_id } = req.params;
  const query = `call project.get_Infant(${parseInt(user_id)});`;
  console.log(query);
  db.query(query, (err, result) => {
    if (err) {
      console.log(err, "error");
    } else {
    //   console.log(result[0], "result");
      res.send(result[0]);
    }
  });
});

router.delete("/deleteInfant/:id", (req, res) => {
    console.log(req.params, "id");
    const { id } = req.params;
    const query = `call project.delete_infant(${parseInt(id)});`;
    console.log(query);
    db.query(query, (err, result) => {
      if (err) {
        console.log(err, "error");
      } else {
        console.log(result[0], "result");
        res.send(result[0]);
      }
    });
  });

  router.get("/get_infant_prediction/:infant_id/:guardian_id", (req, res) => {
    console.log(req.params, "user_id");
    const { infant_id,guardian_id } = req.params;
    const query = `call project.get_Prediction(${parseInt(infant_id)},${parseInt(guardian_id)});`;
    console.log(query);
    db.query(query, (err, result) => {
      if (err) {
        console.log(err, "error");
      } else {
        console.log(result[0], "result");
        res.send(result[0]);
      }
    });
  });

module.exports = router;
