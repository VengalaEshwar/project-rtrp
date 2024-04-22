const router = require("./routes/routes.js");
const express = require('express');

const app = express();

app.listen(5000,()=>{
    console.log("Server is started");
})

app.use(express.json());
router.route(app);
