const Router=require("express");

const db=require("../database.js")

const { exec } = require('child_process');

const axios = require('axios');

const router=Router();

router.post("/processData", (req, res) => {
    const arr=["Healthy","Sick"];
    const {
        infant_id,
        user_id,
        Age_Months,
        Weight_Kg,
        Height_Cm,
        Oxygen_Saturation,
        Pulse_Rate,
        Temperature_C,
        Fever,
        Respiratory_Rate,
        Cough,
        Runny_Nose,
        Skin_Rash,
        Vomiting,
        Diarrhea,
        Blood_Pressure,
        Sleep_Duration_Hrs,
        Feeding_Method,
        Immunization_Status,
        Hygiene_Score,
        Parental_Education,
        Family_Income
    } = req.body;
    const pythonScript = 'Model.py';

    // Execute the Python script with arguments
    const command = `python ${pythonScript} "${Age_Months}" "${Weight_Kg}" "${Height_Cm}" "${Oxygen_Saturation}" "${Pulse_Rate}" "${Temperature_C}" "${Fever}" "${Respiratory_Rate}" "${Cough}" "${Runny_Nose}" "${Skin_Rash}" "${Vomiting}" "${Diarrhea}" "${Blood_Pressure}" "${Sleep_Duration_Hrs}" "${Feeding_Method}" "${Immunization_Status}" "${Hygiene_Score}" "${Parental_Education}" "${Family_Income}"`;

    const pythonProcess = exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing Python script: ${error}`);
            res.status(500).json({ error: 'Something went wrong on the server.' });
            return;
        }
        const output = JSON.parse(stdout);
        console.log('Prediction', output[0]);
        const prediction=arr[output[0]];
        console.log(prediction)
        const query=`call project.prediction_data(${Oxygen_Saturation}, ${Pulse_Rate}, ${parseFloat(Temperature_C)},${Fever}, ${Respiratory_Rate}, ${parseInt(Cough)},${parseInt(Runny_Nose)}, ${parseInt(Skin_Rash)},${parseInt(Vomiting)},${parseInt(Diarrhea)}, '${Blood_Pressure}', ${parseInt(Hygiene_Score)}, '${Parental_Education}', ${parseInt(Family_Income)}, '${prediction}', ${parseInt(infant_id)}, ${parseInt(user_id)});`;
        console.log(query);
        db.query(query,(err,result)=>{
            if (err) {
                console.log(err, "error");
              } else {
                console.log(result[0][0].msg, "result");
                res.json({ Prediction: output[0],
                    message:result[0][0].msg
                });
              }
        })
     
    });
});


module.exports=router;