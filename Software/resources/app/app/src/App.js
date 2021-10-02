import classNames from "classnames";
import React, { Component, useState } from "react";

const INPUTS_NUM = [
    { label: "Epochs", key: "epochs" },
    { label: "Batch Size", key: "batch_size" },
    { label: "LR", key: "lr" },
    { label: "Neurons", key: "neurons" },
    { label: "Starting Index", key: "starting_index" },
];
const NUM_INPUTS = INPUTS_NUM.map((i) => i.key);

const INPUTS_2 = [
    { label: "Optimizer", key: "optimizer" },
    { label: "Kernel Init", key: "kernel_init" },
    { label: "Bias Init", key: "bias_init" },
];

const INPUTS = [...INPUTS_NUM, ...INPUTS_2];

const YEARS = [
    { label: "Starting Year", key: "start_year" },
    { label: "Test Year Start", key: "test_start_year" },
    { label: "Ending Year", key: "end_year" },
];

const VARIABLES = [
    { label: "A", key: "a" },
    { label: "S", key: "s" },
    { label: "Ele", key: "ele" },
    { label: "Sp", key: "sp" },
    { label: "MeTO", key: "meto" },
    { label: "MeLE", key: "mele" },
    { label: "MeCO", key: "meco" },
    { label: "MAT", key: "mat" },
    { label: "MAP", key: "map" },
    { label: "Bamboo", key: "bamboo" },
];

const Buttons = ({ value, onClick, label, className, disabled }) => {
    let allClasses = classNames("location-button", {
        disabled: disabled,
    });
    return (
        <div className={className}>
            <button
                disabled={disabled}
                className={allClasses}
                onClick={onClick}
            >
                {label}
            </button>
            {value && <label className="location-name">Current: {value}</label>}
        </div>
    );
};

const Table = ({ data, type, onChange, values }) => {
    const onValChange = (event, key) => {
        if (type === "checkbox") {
            return onChange(event.target.checked, key);
        }
        return onChange(event.target.value, key);
    };
    const label = data.map((datum) => {
        return (
            <th key={datum.label}>
                <label>{datum.label}</label>
            </th>
        );
    });

    const keys = data.map((datum) => {
        const value = values && values[datum.key];
        const valueProp =
            type === "checkbox" ? { checked: value } : { value: value };
        return (
            <td key={datum.key}>
                <input
                    type={type}
                    onChange={(e) => onValChange(e, datum.key)}
                    {...valueProp}
                />
            </td>
        );
    });

    return (
        <table className="table">
            <tbody>
                <tr>{label}</tr>
                <tr>{keys}</tr>
            </tbody>
        </table>
    );
};

const TreeSplit = ({ treeData, onChange }) => {
    const [selectedTrainingTrees, setSelectedTrainingTrees] = useState(
        new Set()
    );
    const [selectedTestingTrees, setSelectedTestingTrees] = useState(new Set());

    const onClickTree = (name, type) => {
        if (type === "TRAINING") {
            setSelectedTrainingTrees((prevValue) => {
                if (prevValue.has(name)) {
                    prevValue.delete(name);
                } else {
                    prevValue.add(name);
                }
                return new Set(prevValue);
            });
        } else {
            setSelectedTestingTrees((prevValue) => {
                if (prevValue.has(name)) {
                    prevValue.delete(name);
                } else {
                    prevValue.add(name);
                }
                return new Set(prevValue);
            });
				}
		};
		
		const moveSelectedTrees = (moveTo) => {
			const currentTrainingTrees = new Set(treeData.filter(({ type }) => type === 'TRAINING').map(({ name}) => name));
			const currentTestingTrees = new Set(treeData.filter(({ type }) => type === 'TESTING').map(({ name}) => name));
			
			if (moveTo === "TRAINING") {
				Array.from(selectedTestingTrees).forEach((name) => {
					currentTestingTrees.delete(name);
					currentTrainingTrees.add(name);
				})
			} else {
				Array.from(selectedTrainingTrees).forEach((name) => {
					currentTrainingTrees.delete(name);
					currentTestingTrees.add(name);
				})
			}

			const newTestingTrees = Array.from(currentTestingTrees).map(name => ({ name, type: 'TESTING' }));
			const newTrainingTrees = Array.from(currentTrainingTrees).map(name => ({ name, type: 'TRAINING' }));

			setSelectedTrainingTrees(new Set())
			setSelectedTestingTrees(new Set())
			onChange(newTestingTrees.concat(newTrainingTrees));
		}

    const trainTrees = treeData.filter(({ type }) => {
        return type === "TRAINING";
    });
    const testTrees = treeData.filter(({ type }) => {
        return type === "TESTING";
    });

    return (
        <div className='tree-split-container flex-horiz'>
            <div className='tree-split-training-container'>
                {trainTrees.map(({ name }) => {
                    return (
                        <div
														key={name}
                            className={classNames('tree-split-tree-item', 'pointer-cursor', {
                                "tree-split-is-selected": selectedTrainingTrees.has(name),
                            })}
                            onClick={() => onClickTree(name, "TRAINING")}
                        >
                            {name}
                        </div>
                    );
                })}
            </div>
            <div className='flex-vert tree-split-op-container'>
							<button className='tree-split-op' onClick={() => moveSelectedTrees("TRAINING")}>{"<"}</button>
							<button className='tree-split-op' onClick={() => moveSelectedTrees("TESTING")}>{">"}</button>
						</div>
            <div className='tree-split-testing-container'>
                {testTrees.map(({ name }) => {
                    return (
                        <div
														key={name}
                            className={classNames('tree-split-tree-item', 'pointer-cursor', {
                                "tree-split-is-selected": selectedTestingTrees.has(name),
                            })}
                            onClick={() => onClickTree(name, "TESTING")}
                        >
                            {name}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

class App extends Component {
    constructor(props) {
        super(props);
        const state = {
            inputs: {},
            variables: {},
            years: {},
            dataset_path: "",
            results_path: "",
            train_script_path: "",
            train_script_path_tree_split: "",
            predict_script_path: "",
						hyperparameters_path: "",
						treeSplitData: [],
            splitByTree: false,
        };

        INPUTS.forEach((input) => {
            state.inputs[input.key] = "";
        });

        VARIABLES.forEach((variable) => {
            state.variables[variable.key] = false;
        });

        YEARS.forEach((year) => {
            state.years[year.key] = "";
        });

        this.state = state;
    }

    async fetchInitData() {
        let data;
        try {
            const response = await fetch("http://localhost:3001/init", {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
            });
            data = await response.json();
        } catch (err) {
            console.error("error", err);
            return;
        }

        const state = {
            inputs: {},
            variables: {},
						years: {},
						tree_split_data: [],
				};
				
				const allTreeNames = data.tree_names || [];
				const trainTestSplitTrees = data.train_test_tree_split || [];
				const testTrees = new Set(trainTestSplitTrees[1] || []);

				state.treeSplitData = allTreeNames.map(treeName => {
					if (testTrees.has(treeName)) {
						return { name: treeName, type: 'TESTING' };
					}
					return { name: treeName, type: 'TRAINING' };
				})

        INPUTS.forEach((input) => {
            state.inputs[input.key] = data[input.key];
        });

        const variablesArr = data.variables || [];
        VARIABLES.forEach((variable) => {
            state.variables[variable.key] = variablesArr.includes(
                variable.label
            );
        });

        const yearsArr = data.train_test_split || [];
        YEARS.forEach((year, index) => {
            state.years[year.key] = yearsArr[index];
        });

        this.setState({
            dataset_path: data.dataset_path,
            results_path: data.results_path,
            hyperparameters_path: data.hyperparameters_path,
            train_script_path: data.train_script_path,
            train_script_path_tree_split: data.train_script_path_tree_split,
            predict_script_path: data.predict_script_path,
            ...state,
        });
    }

    onButton = (key) => {
        if (
            key === "dataset_path" ||
            key === "results_path" ||
            key === "hyperparameters_path" ||
            key === "train_script_path" ||
            key === "train_script_path_tree_split" ||
            key === "predict_script_path"
        ) {
            this.onOpenFile(key);
        } else if (key === "run_training") {
            this.onRunTraining();
        } else if (key === "run_training_tree_split") {
			this.onRunTrainingTreeSplit();
		} else if (key === "run_prediction") {
            this.onRunPrediction();
        } else if (key === "save_parameters") {
            this.onSaveParameters();
        } else if (key === "set_environment") {
            this.onSetEnvironment();
        }
    };

    get getParameters() {
        const {
            inputs,
            variables,
            years,
            dataset_path,
            results_path,
						train_script_path,
						train_script_path_tree_split,
						predict_script_path,
						treeSplitData,
        } = this.state;

        const variablesArr = [];
        VARIABLES.forEach((varObj) => {
            if (variables[varObj.key]) {
                variablesArr.push(varObj.label);
            }
        });

        const yearsArr = [];
        YEARS.forEach((yearObj) => {
            const val = years[yearObj.key];
            yearsArr.push(val);
				});
				
				const train_test_tree_split = [[], []];
				treeSplitData.forEach(({ name, type }) => {
					if (type === 'TRAINING') {
						train_test_tree_split[0].push(name);
					} else if (type === 'TESTING') {
						train_test_tree_split[1].push(name);
					}
				})

        return {
            ...inputs,
            train_test_split: yearsArr,
            variables: variablesArr,
            dataset_path,
            results_path,
						train_script_path,
						train_script_path_tree_split,
						predict_script_path,
						train_test_tree_split,
        };
    }

    async sendRequest(url, mode, params) {
        let data;
        let response;
        try {
            response = await fetch(url, {
                method: mode,
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(params),
            });
            data = response.json();
        } catch (err) {
            console.error("error", err);
            return;
        }
        return data;
    }

    async onSaveParameters() {
        const data = await this.sendRequest(
            "http://localhost:3001/saveParameters",
            "POST",
            { parameters: this.getParameters }
        );
        if (data) {
            console.log("Saving Parameters");
        }
    }

    async onSetEnvironment() {
        const data = await this.sendRequest(
            "http://localhost:3001/setEnvironment",
            "POST",
            {}
        );
        if (data) {
            console.log("Setting Conda Environment");
        }
    }

    async onRunPrediction() {
        const data = await this.sendRequest(
            "http://localhost:3001/runPrediction",
            "POST",
            { parameters: this.getParameters }
        );
        if (data) {
            console.log("Running Prediction");
        }
    }

    async onRunTraining() {
        const data = await this.sendRequest(
            "http://localhost:3001/runTraining",
            "POST",
            { parameters: this.getParameters }
        );
        if (data) {
            console.log(data);
        }
		}

    async onRunTrainingTreeSplit() {
        const data = await this.sendRequest(
            "http://localhost:3001/runTrainingTreeSplit",
            "POST",
            { parameters: this.getParameters }
        );
        if (data) {
            console.log(data);
        }
    }

    async onOpenFile(key) {
        const data = await this.sendRequest(
            "http://localhost:3001/openFile",
            "POST",
            { path: key }
        );
        if (data) {
			this.setState({ [key]: data });
        }
        if (key === "hyperparameters_path" || key === "dataset_path") {
            this.fetchInitData();
        }
    }

    onInputChange = (val, key) => {
        let parsedVal = val;
        if (NUM_INPUTS.includes(key)) {
            parsedVal = Number(val);
            if (isNaN(parsedVal)) {
                return;
            }
        }
        this.onChange(parsedVal, key, "inputs");
    };

    onSplitTypeChange = (event) => {
        this.setState({ splitByTree: event.target.value === "SPLIT_TREE" });
    };

    onYearsChange = (val, key) => {
        const valNum = Number(val);
        if (isNaN(valNum)) {
            return;
        }
        this.onChange(valNum, key, "years");
		};
		
		onTreeSplitDataChange = (newData) => {
			this.setState({ treeSplitData: newData });
		}

    onVariablesChange = (val, key) => {
        this.onChange(val, key, "variables");
    };

    onChange = (val, key, field) => {
        this.setState((prevState) => {
            return {
                [field]: {
                    ...prevState[field],
                    [key]: val,
                },
            };
        });
    };

    render() {
        return (
            <div className="App">
                <h1 id="title">Predicting Tree Growth Algorithm</h1>
                <div className='section'>
                    <h2>Setup</h2>
                    <ol>
                        <li>
                            Install Anaconda (not miniconda)
                            <ul>
                                <li>https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html</li>
                                <li>Check "Add to PATH" during installation</li>
                                <li>Restart this software after installation</li>
                            </ul>
                        </li>
                        <li>
                            IMPORTANT: Make sure there are no spaces in the paths to any of the files
                            <ul>
                                <li>
                                    Tip: Save it in a root directory (C:/)
                                </li>
                            </ul>
                        </li>
                    </ol>
                </div>
                <div className="section">
                    <h2>Locations</h2>
                    <Buttons
                        value={this.state.hyperparameters_path}
                        onClick={() => this.onButton("hyperparameters_path")}
                        label="Choose Hyperparameters File"
                    />
                    <Buttons
                        value={this.state.dataset_path}
                        onClick={() => this.onButton("dataset_path")}
                        label="Choose Dataset File"
                    />
                    <Buttons
                        value={this.state.results_path}
                        onClick={() => this.onButton("results_path")}
                        label="Choose Results Directory"
                    />
                    <Buttons
                        value={this.state.train_script_path}
                        onClick={() => this.onButton("train_script_path")}
                        label="Choose Training Script File"
                    />
                    <Buttons
                        value={this.state.train_script_path_tree_split}
                        onClick={() => this.onButton("train_script_path_tree_split")}
                        label="Choose Training Script File using Tree-Splitting"
                    />
                    <Buttons
                        value={this.state.predict_script_path}
                        onClick={() => this.onButton("predict_script_path")}
                        label="Choose Prediction Script Directory"
                    />
                </div>
                <div className="section">
                    <h2>Inputs</h2>
                    <Table
                        data={INPUTS_NUM}
                        type="text"
                        onChange={this.onInputChange}
                        values={this.state.inputs}
                    />
                    <Table
                        data={INPUTS_2}
                        type="text"
                        onChange={this.onInputChange}
                        values={this.state.inputs}
                    />

                    <Table
                        data={VARIABLES}
                        type="checkbox"
                        onChange={this.onVariablesChange}
                        values={this.state.variables}
                    />
                </div>
                <div className="section">
                    <div className="flex-horiz">
                        <label className="padding-right">Split type:</label>
                        <div className="radio padding-right">
                            <label>
                                <input
                                    type="radio"
                                    value="SPLIT_YEAR"
                                    checked={this.state.splitByTree === false}
                                    onChange={this.onSplitTypeChange}
                                />
                                Split by Year
                            </label>
                        </div>
                        <div className="radio padding-right">
                            <label>
                                <input
                                    type="radio"
                                    value="SPLIT_TREE"
                                    checked={this.state.splitByTree === true}
                                    onChange={this.onSplitTypeChange}
                                />
                                Split by Tree
                            </label>
                        </div>
                    </div>
                    <div className="content">
                        {this.state.splitByTree ? (
													<TreeSplit treeData={this.state.treeSplitData} onChange={this.onTreeSplitDataChange} />
												) : (
                            <Table
                                data={YEARS}
                                type="text"
                                onChange={this.onYearsChange}
                                values={this.state.years}
                            />
                        )}
                    </div>
                </div>
                <div className="section">
                    <h2>Actions</h2>
                    <Buttons
                        onClick={() => this.onButton("set_environment")}
                        label="Set-up Environment"
                        className="actions"
                    />
                    <Buttons
                        onClick={() => this.onButton("save_parameters")}
                        label="Save Parameters"
                        className="actions"
                        disabled={this.state.hyperparameters_path === ""}
                    />
                    <Buttons
                        onClick={() => this.onButton("run_training")}
                        label="Run Training"
                        className="actions"
                        disabled={this.state.train_script_path === ""}
                    />
                    <Buttons
                        onClick={() => this.onButton("run_training_tree_split")}
                        label="Run Training Using Tree-Split"
                        className="actions"
                        disabled={this.state.train_script_path_tree_split === ""}
                    />
                    <Buttons
                        onClick={() => this.onButton("run_prediction")}
                        label="Run Prediction"
                        className="actions"
                        disabled={this.state.predict_script_path === ""}
                    />
                </div>
            </div>
        );
    }
}

export default App;
