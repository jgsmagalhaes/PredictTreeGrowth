// Basic init
const { app, BrowserWindow, dialog } = require('electron')
const csvParser = require('csv-parser');
const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const { spawn, exec } = require('child_process');
const url = require('url');

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let win

function createWindow () {
  // Create the browser window.
  win = new BrowserWindow({ width: 1100, height: 800, webPreferences: { nodeIntegration: true, contextIsolation: false } })

  // and load the index.html of the app.
	win.loadURL(`file://${__dirname}/app/index.html`);

  // Emitted when the window is closed.
  win.on('closed', () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    win = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (win === null) {
    createWindow()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.

// Variables
const bashScript = path.resolve(path.join(__dirname, 'bashScript.sh'));
const batchScript = path.resolve(path.join(__dirname, 'batchScript.bat'));
let trainingScriptPath = '';
let treeSplitTrainingPath = '';
let predictionScriptPath = '';
let hyperparametersPath = '';
let hyperParameters = {};
let datasetPath;
let resultsPath;
let treeSplit = [];

let hyperparametersPathBak = hyperparametersPath + '_bak';
let resultsPathBak = resultsPath + '_bak';

function loadHyperparameters() {
	hyperParameters = fs.readFileSync(hyperparametersPath, 'utf-8');
	try {
		hyperParameters = JSON.parse(hyperParameters);
	} catch(err) {
		hyperParameters = {};
	}
	datasetPath = path.resolve(hyperParameters.dataset_path);
	resultsPath = path.resolve(hyperParameters.results_path);
	trainingScriptPath = path.resolve(hyperParameters.train_script_path);
	treeSplitTrainingPath = path.resolve(hyperParameters.train_script_path_tree_split);
	predictionScriptPath = path.resolve(hyperParameters.predict_script_path);
	treeSplit = hyperParameters.train_test_tree_split;

	hyperparametersPathBak = hyperparametersPath + '_bak';
	resultsPathBak = resultsPath + '_bak';
}

function getTreeNames() {
	return new Promise((resolve, reject) => {
		if (!datasetPath || !fs.existsSync(datasetPath)) {
			reject("Invalid dataset path");
			return;
		}
		const treeNames = new Set();
		fs.createReadStream(datasetPath)
			.on("error", () => {
				reject("Error reading file")
			})
			.pipe(csvParser())
			.on("data", (row) => {
				const tree = row["TREE"];
				treeNames.add(tree);
			})
			.on("end", () => {
				resolve(Array.from(treeNames));
			})
	})
}

function saveHyperparameters(parameters) {
	const newParameters = parameters;
	if (newParameters.variables) {
		const newVariables = ['TREE', 'BAI'];
		newParameters.variables.forEach(v => newVariables.push(v));
		newParameters.variables = newVariables;
	}
	try {
		fs.renameSync(hyperparametersPath, hyperparametersPathBak);
		fs.writeFileSync(hyperparametersPath, JSON.stringify(parameters, null, 2));
	} catch(error) {
		return { error };
	}
	return {};
}

function setEnvironment() {
	const commands = [
		`cd "${__dirname}"`,
		'conda env create -f environment.yml',
	]
	openTerminal(commands.join(' && '));
	return commands.join(' && ');
}

function runTraining() {
	const trainingDir = path.dirname(trainingScriptPath);
	const trainingFilename = path.basename(trainingScriptPath);
	const commands = [
		`${process.platform === 'darwin' ? 'source activate' : 'activate'} predict_bai`,
		`cd "${trainingDir}"`,
		`python ${trainingFilename} --hyperparameters="${hyperparametersPath}"`,
	]
	const command = commands.join(' && ');
	openTerminal(command);
}

function runTrainingTreeSplit() {
	const trainingDir = path.dirname(treeSplitTrainingPath);
	const trainingFilename = path.basename(treeSplitTrainingPath);
	const commands = [
		`${process.platform === 'darwin' ? 'source activate' : 'activate'} predict_bai`,
		`cd "${trainingDir}"`,
		`python ${trainingFilename} --hyperparameters="${hyperparametersPath}"`,
	]
	const command = commands.join(' && ');
	openTerminal(command);
}

function runPrediction() {
	const predictionDir = path.dirname(predictionScriptPath);
	const predictionFilename = path.basename(predictionScriptPath);
	const commands = [
		`${process.platform === 'darwin' ? 'source activate' : 'activate'} predict_bai`,
		`cd "${predictionDir}"`,
		`python ${predictionFilename} --hyperparameters="${hyperparametersPath}"`,
	]
	const command = commands.join(' && ');
	openTerminal(command);
}

function openTerminal(command) {
	if (process.platform === 'darwin') {
  	const script = '#!/bin/bash\n' + command;
		fs.writeFileSync(bashScript, script);
		let openTerminalAtPath = spawn ('open', [ '-a', 'Terminal', bashScript ]);
		openTerminalAtPath.stdout.on('data', function(data) {
      console.log(data.toString());
	  });
		openTerminalAtPath.stderr.on('data', function(data) {
			console.log(data.toString());
		});
	} else if (process.platform === 'win32') {

    const script = command;
    fs.writeFileSync(batchScript, script);
    exec(`start cmd.exe /K "${batchScript}"`);
    // let openTerminalAtPath = spawn ('cmd', ['/C', batchScript]);
    // openTerminalAtPath.stdout.on('data', function(data) {
    //   console.log(data.toString());
    // });
    // openTerminalAtPath.stderr.on('data', function(data) {
    //   console.log(data.toString());
    // });
  }
}

const apiApp = express();
apiApp.use(bodyParser.urlencoded({ extended: false }))
apiApp.use(bodyParser.json())
apiApp.use(cors())


// HTTP Routes
apiApp.get('/init', async (req, res) => {
	const params = hyperParameters;
	params.dataset_path = datasetPath;
	params.results_path = resultsPath;
	params.hyperparameters_path = hyperparametersPath;
	params.train_script_path = trainingScriptPath;
	params.train_script_path_tree_split = treeSplitTrainingPath;
	params.predict_script_path = predictionScriptPath;
	params.train_test_tree_split = treeSplit;
	params.tree_names = await getTreeNames();
	res.status(200).json(params);
})

const OPEN_FILE_PATHS = ['dataset_path', 'hyperparameters_path', 'train_script_path', 'train_script_path_tree_split', 'predict_script_path'];
apiApp.post('/openFile', (req, res) => {
	const pathType = req.body.path;
	dialog.showOpenDialog({
    title:"Select " + pathType,
    properties: [OPEN_FILE_PATHS.includes(pathType) ? 'openFile' : 'openDirectory'],
	}).then(({filePaths}) => {
    if (filePaths === undefined){
      res.status(500).json({ error: 'No file selected' })
      return;
    } else {
			const filePath = filePaths[0];
			if (pathType === 'dataset_path') {
				datasetPath = filePath;
			} else if (pathType === 'results_path') {
				resultsPath = filePath;
			} else if (pathType === 'train_script_path') {
				trainingScriptPath = filePath;
			} else if (pathType === 'train_script_path_tree_split') {
				treeSplitTrainingPath = filePath;
			} else if (pathType === 'predict_script_path') {
				predictionScriptPath = filePath;
			} else if (pathType === 'hyperparameters_path') {
				hyperparametersPath = filePath;
				loadHyperparameters();
			}
      res.status(200).json(filePath)
    }
	});
});

apiApp.post('/setEnvironment', (req, res) => {
	const command = setEnvironment();
	console.log('setEnvironment');
	res.status(200).json({ status: 'Completed', command });
});

apiApp.post('/saveParameters', (req, res) => {
	const result = saveHyperparameters(req.body.parameters);
	if (result.error) {
		res.status(500).json(result.error);
	}
	console.log('saveParameters');
	res.status(200).json({ status: 'Completed' });
});

apiApp.post('/runPrediction', (req, res) => {
	const result = saveHyperparameters(req.body.parameters);
	if (result.error) {
		res.status(500).json(result.error);
	}
	runPrediction();
	res.status(200).json('Running Training');
});

apiApp.post('/runTraining', (req, res) => {
	const result = saveHyperparameters(req.body.parameters);
	if (result.error) {
		res.status(500).json(result.error);
	}
	runTraining();
	res.status(200).json('Running Training');
	console.log('runTraining');
});

apiApp.post('/runTrainingTreeSplit', (req, res) => {
	const result = saveHyperparameters(req.body.parameters);
	if (result.error) {
		res.status(500).json(result.error);
	}
	runTrainingTreeSplit();
	res.status(200).json('Running Training using Tree-Splitting');
	console.log('runTrainingTreeSplit');
});

apiApp.listen(3001)
console.log('Running on port 3001')
