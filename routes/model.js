const Router=require("express");

const db=require("../database.js")

const { exec } = require('child_process');

const axios = require('axios');

const router=Router();

// Function to call the API with updated values
// const processDataWithPython = async () => {
//     var oxygen = 12;
//     var pulserate = 23;
//     var temperature = 43;
  
//     try {
//       const response = await axios.get(`http://localhost:3000/api/auth/process-data/${oxygen}/${pulserate}/${temperature}`);
//       const pythonScript = 'App.py';
//       const command = `python ${pythonScript} ${oxygen} ${pulserate} ${temperature}`;
  
//       exec(command, (error, stdout, stderr) => {
//         if (error) {
//           console.error(`Error executing Python script: ${error}`);
//           // Handle the error...
//         } else {
//           try {
//             const result = JSON.parse(stdout);
//             console.log('API Response:', result);
//             // Do something with the result...
//           } catch (parseError) {
//             console.error('Error parsing Python script output:', parseError);
//             // Handle the parsing error...
//           }
//         }
//       });
//     } catch (error) {
//       console.error('Error calling API:', error.message);
//       // Handle the API call error...
//     }
//   };
  
//   // Interval in milliseconds (1 second)
//   const interval = 1000;
  
//   // IIFE to call the function with setInterval
//   (async () => {
//     await processDataWithPython(); // Call the function immediately
//     setInterval(processDataWithPython, interval);
//   })();

// router.get('/process-data/:oxygen/:pulserate/:temperature', (req, res) => {
//     const { oxygen, pulserate, temperature } = req.params;
//     console.log(oxygen, pulserate,temperature)
//     const pythonScript = 'App.py';
//     const pythonProcess = exec(`python ${pythonScript} ${oxygen} ${pulserate} ${temperature}`, (error, stdout, stderr) => {
//       if (error) {
//         console.error(`Error executing Python script: ${error}`);
//         res.status(500).json({ error: 'Something went wrong on the server.' });
//         return;
//       }
//       const result = JSON.parse(stdout);
//       console.log("Fahad", result)
//       res.send({val:result});
//     });
//   });


// router.get("/processData/:Age_Months/:Weight_Kg/:Height_Cm/:Oxygen_Saturation/:Pulse_Rate/:Temperature_C/:Fever/:Respiratory_Rate/:Cough/:Runny_Nose/:Skin_Rash/:Vomiting/:Diarrhea/:Blood_Pressure/:Sleep_Duration_Hrs/:Feeding_Method/:Immunization_Status/:Hygiene_Score/:Parental_Education/:Family_Income", (req, res) => {
//     const {
//         Age_Months,
//         Weight_Kg,
//         Height_Cm,
//         Oxygen_Saturation,
//         Pulse_Rate,
//         Temperature_C,
//         Fever,
//         Respiratory_Rate,
//         Cough,
//         Runny_Nose,
//         Skin_Rash,
//         Vomiting,
//         Diarrhea,
//         Blood_Pressure,
//         Sleep_Duration_Hrs,
//         Feeding_Method,
//         Immunization_Status,
//         Hygiene_Score,
//         Parental_Education,
//         Family_Income
//     } = req.params;

//     console.log(Age_Months);
//     console.log(Weight_Kg);
//     console.log(Height_Cm);
//     console.log(Oxygen_Saturation);
//     console.log(Pulse_Rate);
//     console.log(Temperature_C);
//     console.log(Fever);
//     console.log(Respiratory_Rate);
//     console.log(Cough);
//     console.log(Runny_Nose);
//     console.log(Skin_Rash);
//     console.log(Vomiting);
//     console.log(Diarrhea);
//     console.log(Blood_Pressure);
//     console.log(Sleep_Duration_Hrs);
//     console.log(Feeding_Method);
//     console.log(Immunization_Status);
//     console.log(Hygiene_Score);
//     console.log(Parental_Education);
//     console.log(Family_Income);

//     // Modify the Python script path accordingly
//     const pythonScript = 'Model.py';

//     // Execute the Python script with arguments
//     const command = `python ${pythonScript} "${Age_Months}" "${Weight_Kg}" "${Height_Cm}" "${Oxygen_Saturation}" "${Pulse_Rate}" "${Temperature_C}" "${Fever}" "${Respiratory_Rate}" "${Cough}" "${Runny_Nose}" "${Skin_Rash}" "${Vomiting}" "${Diarrhea}" "${Blood_Pressure}" "${Sleep_Duration_Hrs}" "${Feeding_Method}" "${Immunization_Status}" "${Hygiene_Score}" "${Parental_Education}" "${Family_Income}"`;

//     const pythonProcess = exec(command, (error, stdout, stderr) => {
//         if (error) {
//             console.error(`Error executing Python script: ${error}`);
//             res.status(500).json({ error: 'Something went wrong on the server.' });
//             return;
//         }
//         const result = JSON.parse(stdout);
//         console.log('Prediction', result[0]);
//         res.json({ Prediction: result[0] });
//     });
// });

router.get('/getData',(req,res)=>{

       console.log("RESULT")
     
     
     res.send({msg:"Fahad"})
});



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

    console.log(Age_Months);
    console.log(Weight_Kg);
    console.log(Height_Cm);
    console.log(Oxygen_Saturation);
    console.log(Pulse_Rate);
    console.log(Temperature_C);
    console.log(Fever);
    console.log(Respiratory_Rate);
    console.log(Cough);
    console.log(Runny_Nose);
    console.log(Skin_Rash);
    console.log(Vomiting);
    console.log(Diarrhea);
    console.log(Blood_Pressure);
    console.log(Sleep_Duration_Hrs);
    console.log(Feeding_Method);
    console.log(Immunization_Status);
    console.log(Hygiene_Score);
    console.log(Parental_Education);
    console.log(Family_Income);
    console.log(infant_id);
    console.log(user_id);
    // Modify the Python script path accordingly
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