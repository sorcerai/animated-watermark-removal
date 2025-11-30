/**
 * ComfyUI Client (Node.js Wrapper)
 *
 * Spawns Python subprocess to communicate with ComfyUI
 * Provides TypeScript interface for video processing
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import path from 'path';

export interface ComfyUIOptions {
    host?: string;
    pythonPath?: string;
}

export interface ProcessVideoOptions {
    videoPath: string;
    samPoint: [number, number];
    workflowPath?: string;
    onProgress?: (progress: number) => void;
}

export interface ProcessVideoResult {
    promptId: string;
    status: 'success' | 'failed';
    outputFrames: any;
    processingTime: number;
}

export class ComfyUIClient extends EventEmitter {
    private host: string;
    private pythonPath: string;
    private clientPath: string;

    constructor(options: ComfyUIOptions = {}) {
        super();

        this.host = options.host || 'http://localhost:8188';
        this.pythonPath = options.pythonPath || 'python';
        this.clientPath = path.join(__dirname, 'client.py');
    }

    /**
     * Process video through ComfyUI workflow
     *
     * @param options Processing options
     * @returns Promise resolving to processing result
     */
    async processVideo(options: ProcessVideoOptions): Promise<ProcessVideoResult> {
        const {
            videoPath,
            samPoint,
            workflowPath = 'workflows/templates/sora-removal-production.json',
            onProgress
        } = options;

        return new Promise((resolve, reject) => {
            const args = [
                this.clientPath,
                videoPath,
                JSON.stringify(samPoint)
            ];

            const python: ChildProcess = spawn(this.pythonPath, args, {
                env: {
                    ...process.env,
                    COMFYUI_HOST: this.host,
                    WORKFLOW_PATH: workflowPath
                }
            });

            let output = '';
            let errorOutput = '';

            python.stdout?.on('data', (data: Buffer) => {
                const text = data.toString();

                // Check for progress updates
                const progressMatch = text.match(/Progress: (\d+)%/);
                if (progressMatch && onProgress) {
                    const progress = parseInt(progressMatch[1], 10);
                    onProgress(progress);
                    this.emit('progress', progress);
                }

                // Accumulate output for final JSON parse
                output += text;
            });

            python.stderr?.on('data', (data: Buffer) => {
                errorOutput += data.toString();
                console.error('[ComfyUI Error]:', data.toString());
            });

            python.on('error', (error: Error) => {
                reject(new Error(`Failed to spawn Python: ${error.message}`));
            });

            python.on('close', (code: number | null) => {
                if (code !== 0) {
                    reject(new Error(
                        `ComfyUI processing failed (exit code ${code}):\n${errorOutput}`
                    ));
                    return;
                }

                try {
                    // Parse final JSON output (last line)
                    const lines = output.trim().split('\n');
                    const resultLine = lines[lines.length - 1];
                    const result: ProcessVideoResult = JSON.parse(resultLine);

                    this.emit('complete', result);
                    resolve(result);
                } catch (error) {
                    reject(new Error(
                        `Failed to parse ComfyUI result: ${error}\nOutput: ${output}`
                    ));
                }
            });
        });
    }

    /**
     * Check if ComfyUI server is running
     *
     * @returns Promise resolving to true if running
     */
    async isRunning(): Promise<boolean> {
        try {
            const response = await fetch(`${this.host}/system_stats`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    /**
     * Get system stats from ComfyUI
     *
     * @returns System stats including VRAM usage
     */
    async getSystemStats(): Promise<any> {
        const response = await fetch(`${this.host}/system_stats`);
        if (!response.ok) {
            throw new Error(`Failed to get system stats: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Get current queue status
     *
     * @returns Queue information
     */
    async getQueue(): Promise<any> {
        const response = await fetch(`${this.host}/queue`);
        if (!response.ok) {
            throw new Error(`Failed to get queue: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Cancel all pending jobs in queue
     */
    async clearQueue(): Promise<void> {
        const response = await fetch(`${this.host}/queue`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ clear: true })
        });

        if (!response.ok) {
            throw new Error(`Failed to clear queue: ${response.statusText}`);
        }
    }
}

// Example usage
if (require.main === module) {
    const client = new ComfyUIClient();

    client.processVideo({
        videoPath: 'test-video.mp4',
        samPoint: [960, 540],  // 1080p center
        onProgress: (progress) => {
            console.log(`Processing: ${progress}%`);
        }
    })
    .then((result) => {
        console.log('✓ Processing complete:', result);
    })
    .catch((error) => {
        console.error('✗ Processing failed:', error);
        process.exit(1);
    });
}
