import * as vscode from 'vscode';
import * as path from 'path';
import * as cp from 'child_process';
import * as fs from 'fs';

export function activate(context: vscode.ExtensionContext) {
    console.log('Hive-Reflex Extension is active!');

    // Command 1: Compile ONNX Model
    let compileDisposal = vscode.commands.registerCommand('hive-reflex.compileModel', async (uri: vscode.Uri) => {
        if (!uri) {
            vscode.window.showErrorMessage('Please right-click an ONNX file to compile.');
            return;
        }

        const modelPath = uri.fsPath;
        const workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
        if (!workspaceFolder) { return; }

        const config = vscode.workspace.getConfiguration('hive-reflex');
        const compilerScript = config.get<string>('compilerPath') ||
            path.join(workspaceFolder.uri.fsPath, 'mlir_compiler', 'codegen_cim.py');

        // Output paths
        const outputC = modelPath.replace('.onnx', '_gen.c');
        const outputBin = modelPath.replace('.onnx', '_weights.bin');
        const outputConfig = modelPath.replace('.onnx', '_config.json');

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Compiling for Hive-Reflex CIM...",
            cancellable: false
        }, async (progress, token) => {

            return new Promise<void>((resolve, reject) => {
                const cmd = `python "${compilerScript}" --model "${modelPath}" --output-c "${outputC}" --output-weights "${outputBin}" --output-config "${outputConfig}"`;

                cp.exec(cmd, (err, stdout, stderr) => {
                    if (err) {
                        vscode.window.showErrorMessage(`Compilation Failed: ${stderr}`);
                        reject(err);
                    } else {
                        vscode.window.showInformationMessage(`Compilation Success! Generated ${path.basename(outputC)}`);
                        // Automatically show visualization if success
                        vscode.commands.executeCommand('hive-reflex.showScheduler', vscode.Uri.file(outputConfig));
                        resolve();
                    }
                });
            });
        });
    });

    // Command 3: Configure IntelliSense
    let intellisenseDisposal = vscode.commands.registerCommand('hive-reflex.configureIntellisense', async () => {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) { return; }

        const rootPath = workspaceFolders[0].uri.fsPath;
        const vscodeDir = path.join(rootPath, '.vscode');
        if (!fs.existsSync(vscodeDir)) {
            fs.mkdirSync(vscodeDir);
        }

        // Simple generation of c_cpp_properties.json
        const template = {
            "configurations": [{
                "name": "Hive-Reflex SDK",
                "includePath": [
                    "${workspaceFolder}/**",
                    "${workspaceFolder}/imc22_sdk"
                ],
                "defines": ["IMC22_PLATFORM"],
                "cStandard": "c11"
            }],
            "version": 4
        };

        fs.writeFileSync(path.join(vscodeDir, 'c_cpp_properties.json'), JSON.stringify(template, null, 4));
        vscode.window.showInformationMessage('IntelliSense configured for Hive-Reflex SDK!');
    });

    context.subscriptions.push(intellisenseDisposal);

    // Command 2: Visualize Scheduler (Interactive)
    let vizDisposal = vscode.commands.registerCommand('hive-reflex.showScheduler', (uri: vscode.Uri) => {
        const panel = vscode.window.createWebviewPanel(
            'hiveReflexViz',
            'CIM Scheduler Editor',
            vscode.ViewColumn.Two,
            { enableScripts: true }
        );

        const configPath = uri.fsPath;
        let configContent = "{}";
        try { configContent = fs.readFileSync(configPath, 'utf8'); } catch (e) { }

        panel.webview.html = getWebviewContent(configContent);

        // Handle messages from the webview
        panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.command) {
                    case 'toggleLayer':
                        vscode.window.showInformationMessage(`Optimizing Layer ${message.index}: ${message.target.toUpperCase()}...`);

                        // 1. Load existing overrides
                        const overridesPath = configPath.replace('_config.json', '.overrides.json');
                        let overrides: any = {};
                        if (fs.existsSync(overridesPath)) {
                            try {
                                overrides = JSON.parse(fs.readFileSync(overridesPath, 'utf8'));
                            } catch (e) { }
                        }

                        // 2. Update override
                        overrides[message.index] = message.target;
                        fs.writeFileSync(overridesPath, JSON.stringify(overrides, null, 2));

                        // 3. Trigger Re-compilation
                        // We need to find the original ONNX path. Assuming naming convention matches.
                        const onnxPath = configPath.replace('_config.json', '.onnx');

                        if (fs.existsSync(onnxPath)) {
                            vscode.commands.executeCommand('hive-reflex.compileModel', vscode.Uri.file(onnxPath));
                            // Note: Compilation will refresh the config file, but we might need 
                            // to manually refresh the webview content here if we want immediate feedback
                            // For now, let's just let the user re-open or rely on compiler success msg.
                        } else {
                            vscode.window.showErrorMessage("Could not find original ONNX file to recompile.");
                        }
                        return;
                }
            },
            undefined,
            context.subscriptions
        );
    });

    context.subscriptions.push(compileDisposal);
    context.subscriptions.push(vizDisposal);
}

function getWebviewContent(configJson: string) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scheduler Editor</title>
    <style>
        body { font-family: sans-serif; padding: 20px; }
        .layer-box {
            border: 2px solid #ccc;
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.2s;
        }
        .layer-box:hover { transform: scale(1.02); box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        .cim-core { background-color: #dbf0da; border-color: #4CAF50; color: #000; }
        .cpu-core { background-color: #f0dacd; border-color: #F44336; color: #000; }
        .arrow { text-align: center; font-size: 20px; color: #666; }
        h2 { color: #2196F3; }
        .hint { color: #666; font-size: 0.9em; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h2>Heterogeneous Task Graph</h2>
    <div class="hint">👆 Click any layer to toggle between CIM Accelerator and RISC-V CPU</div>
    <div id="graph-container"></div>

    <script>
        const vscode = acquireVsCodeApi();
        const config = ${configJson};
        const container = document.getElementById('graph-container');

        if (config.layers) {
            config.layers.forEach((layer, index) => {
                const div = document.createElement('div');
                div.className = 'layer-box';
                
                // Heuristic or read from config
                let isCim = ['fc', 'lstm', 'softmax', 'layernorm'].includes(layer.type) || (layer.type === 'activation' && layer.activation === 'gelu');
                // Check overrides if they existed
                // if (config.overrides && config.overrides[index]) isCim = config.overrides[index] === 'cim';
                
                div.classList.add(isCim ? 'cim-core' : 'cpu-core');
                
                div.innerHTML = \`
                    <span><strong>Layer \${index}</strong>: \${layer.name} (\${layer.type})</span>
                    <span>\${isCim ? '⚡ CIM Accel' : '🐢 RISC-V CPU'}</span>
                \`;
                
                div.onclick = () => {
                   const newTarget = isCim ? 'cpu' : 'cim';
                   vscode.postMessage({
                       command: 'toggleLayer',
                       index: index,
                       target: newTarget
                   });
                   // Optimistic update
                   div.classList.toggle('cim-core');
                   div.classList.toggle('cpu-core');
                   div.querySelector('span:last-child').innerText = newTarget === 'cim' ? '⚡ CIM Accel' : '🐢 RISC-V CPU';
                   isCim = !isCim;
                };
                
                container.appendChild(div);

                if (index < config.layers.length - 1) {
                    const arrow = document.createElement('div');
                    arrow.className = 'arrow';
                    arrow.innerHTML = '↓';
                    container.appendChild(arrow);
                }
            });
        }
    </script>
</body>
</html>`;
}

export function deactivate() { }
