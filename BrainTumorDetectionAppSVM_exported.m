classdef BrainTumorDetectionAppSVM_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        NameFaiadFaisalSarthokID220021116Label  matlab.ui.control.Label
        BrainTumorDetectionSystemLabel  matlab.ui.control.Label
        ResultPanel                     matlab.ui.container.Panel
        ResultTextArea                  matlab.ui.control.TextArea
        StatusPanel_2                   matlab.ui.container.Panel
        StatusTextArea                  matlab.ui.control.TextArea
        ImageAxesPanel                  matlab.ui.container.Panel
        UIAxes                          matlab.ui.control.UIAxes
        MainPanel                       matlab.ui.container.Panel
        ResetModelButton                matlab.ui.control.Button
        LoadTestImageButton             matlab.ui.control.Button
        TrainModelButton                matlab.ui.control.Button
        LoadDatasetButton               matlab.ui.control.Button
    end

    
       
    properties (Access = private)
        trainingData = struct('loaded', false); 
        trainedModel = struct('trained', false);
    end
    
    methods (Access = private)

    function feat = extractEnhancedFeatures(app, imgPath)
        img = imread(imgPath);

        if size(img,3) == 3, 
            img = rgb2gray(img); 
        end

        img = imresize(img, [128,128]);
        img = double(img)/255;

        brightness = mean(img(:));%Will calculate Average Pixel Intesity
        contrast = std(img(:));%will calculate Standard Deviation of all pixel value

        h = fspecial('average',[5 5]);%for texture rough or smooth
        localMean = imfilter(img,h,'replicate');
        localVar = imfilter((img-localMean).^2,h,'replicate');
        texture = mean(sqrt(localVar(:)));

        edges = edge(img, 'canny', [0.1, 0.3]);%Edge dtetection with Canny Algorithm
        edgeDensity = sum(edges(:)) / numel(edges);

        brightMask = img > 0.8;
        brightRatio = sum(brightMask(:)) / numel(brightMask);%calculate bright ratio

        feat = [brightness, contrast, texture, edgeDensity, brightRatio];
    end
end

    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: LoadDatasetButton
        function LoadDatasetButtonPushed(app, event)
          
            app.StatusTextArea.Value = "Loading dataset...";

            %Open Folder and Select
            datasetPath = uigetdir('', 'Select dataset folder (with "yes" and "no" subfolders)');
            if datasetPath == 0
                app.StatusTextArea.Value = "Cancelled";
                return
            end

            %DefineExpectedFolder
            yesFolder = fullfile(datasetPath, 'yes');
            noFolder  = fullfile(datasetPath, 'no');

            %WllOnlyAcceptFolderWithYesNoSubfolder
            if ~exist(yesFolder, 'dir') || ~exist(noFolder, 'dir')
                uialert(app.UIFigure, 'Dataset must have "yes" and "no" subfolders!', 'Error');
                app.StatusTextArea.Value = "Invalid folder structure";
                return
            end

            %WillCollectAllTypeofImages
            yesFiles = [dir(fullfile(yesFolder, '*.jpg')); dir(fullfile(yesFolder, '*.png')); dir(fullfile(yesFolder, '*.jpeg'))];
            noFiles  = [dir(fullfile(noFolder, '*.jpg')); dir(fullfile(noFolder, '*.png')); dir(fullfile(noFolder, '*.jpeg'))];

            %WillFixMinimumRequirementOf3images
            if length(yesFiles) < 3 || length(noFiles) < 3
                uialert(app.UIFigure, 'Need at least 3 images in each folder!', 'Error');
                app.StatusTextArea.Value = "Not enough images";
                return
            end

            %WillExtractFeaturesfromtheImages
            allFeatures = [];
            allLabels   = [];

            for i = 1:length(yesFiles)
                imgPath = fullfile(yesFolder, yesFiles(i).name);
                feat = app.extractEnhancedFeatures(imgPath);
                allFeatures = [allFeatures; feat];
                allLabels   = [allLabels; 1]; % Label 1 for tumor
            end

            %WillExtractFeaturesfromtheNoTumorImages
            for i = 1:length(noFiles)
                imgPath = fullfile(noFolder, noFiles(i).name);
                feat = app.extractEnhancedFeatures(imgPath);
                allFeatures = [allFeatures; feat];
                allLabels   = [allLabels; 0]; % Label 0 for no tumor
            end

            % StoreTrainingDataInsidetheApp
            app.trainingData.features = allFeatures;
            app.trainingData.labels   = allLabels;
            app.trainingData.loaded   = true;
            
            %WllDisplayHowManyImages
            app.StatusTextArea.Value = sprintf("Data loaded!\nTumor: %d images\nNo tumor: %d images", length(yesFiles), length(noFiles));
        end

        % Button down function: UIAxes
        function UIAxesButtonDown(app, event)
            % Empty function - for future use
        end

        % Button pushed function: TrainModelButton
        function TrainModelButtonPushed(app, event)
               
            %CheckIfDatasetIsLoadedorNot
            if ~app.trainingData.loaded
                uialert(app.UIFigure, 'Load training data first!', 'Error');
                return
            end

            app.StatusTextArea.Value = "Training SVM model...";

            X = app.trainingData.features;
            Y = app.trainingData.labels;

            % Train SVM model
            svmModel = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'Standardize', true, 'BoxConstraint', 1, 'KernelScale', 'auto');

            app.trainedModel.model = svmModel;
            app.trainedModel.trained = true;

           % Split data into 70% train, 30% test
            cv = cvpartition(Y, 'HoldOut', 0.3); 
            Xtrain = X(training(cv), :);
            Ytrain = Y(training(cv));
            Xtest  = X(test(cv), :);
            Ytest  = Y(test(cv));

           % Train model on training data
            svmModel = fitcsvm(Xtrain, Ytrain, ...
            'KernelFunction', 'rbf', ...
            'Standardize', true, ...
            'BoxConstraint', 1, ...
            'KernelScale', 'auto');

           %StoreTrainedModel
            app.trainedModel.model = svmModel;
            app.trainedModel.trained = true;

           % AccuracyOnTrainingSet
            trainPreds = predict(svmModel, Xtrain);
            trainAcc = mean(trainPreds == Ytrain) * 100;

           % AccuracyOnTestSet
            testPreds = predict(svmModel, Xtest);
            testAcc = mean(testPreds == Ytest) * 100;

           % ShowBothResultsOnStatusPanel
            app.StatusTextArea.Value = sprintf("Model trained!\nTrain Accuracy: %.1f%%\nTest Accuracy: %.1f%%", ...
            trainAcc, testAcc);

        end

        % Button pushed function: LoadTestImageButton
        function LoadTestImageButtonPushed(app, event)

            %CheckIfTheModelIsTrainedOrNot
            if ~app.trainedModel.trained
                uialert(app.UIFigure, 'Train the model first!', 'Error');
                return
            end

           %OpenFileToSelectTestingImage
            [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Select test image');
            if file == 0, 
                return; 
            end
            
            %ReadAndShowImage
            imgPath = fullfile(path, file);
            testImg = imread(imgPath);
            imshow(testImg, 'Parent', app.UIAxes);

            %ConvertImagesIntoNumericFeatureVector
            feat = app.extractEnhancedFeatures(imgPath);
            [prediction, ~] = predict(app.trainedModel.model, feat);

            %MakePredictionWithTrainedSVM
            if prediction == 1
                result = 'TUMOR DETECTED';
                col = [1, 0.2, 0.2];%ForShowinginRED
            else
                result = 'NO TUMOR';
                col = [0.2, 0.9, 0.2];%ForShowinginGreen
            end

            %ForisplayingAsTitleofTheImages
            title(app.UIAxes, result, 'Color', col);
            app.ResultTextArea.Value = result;

        end

        % Button pushed function: ResetModelButton
        function ResetModelButtonPushed(app, event)

            app.trainingData = struct('loaded', false);
            app.trainedModel = struct('trained', false);
            cla(app.UIAxes);%ClearAxes
            title(app.UIAxes, 'Image Display');%ResetTitle
            app.ResultTextArea.Value = "Reset complete";
            app.StatusTextArea.Value = "Ready";
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'MATLAB App';

            % Create MainPanel
            app.MainPanel = uipanel(app.UIFigure);
            app.MainPanel.Title = 'Main Panel';
            app.MainPanel.Position = [1 188 152 221];

            % Create LoadDatasetButton
            app.LoadDatasetButton = uibutton(app.MainPanel, 'push');
            app.LoadDatasetButton.ButtonPushedFcn = createCallbackFcn(app, @LoadDatasetButtonPushed, true);
            app.LoadDatasetButton.Position = [26 147 100 22];
            app.LoadDatasetButton.Text = 'Load Dataset';

            % Create TrainModelButton
            app.TrainModelButton = uibutton(app.MainPanel, 'push');
            app.TrainModelButton.ButtonPushedFcn = createCallbackFcn(app, @TrainModelButtonPushed, true);
            app.TrainModelButton.Position = [26 107 100 22];
            app.TrainModelButton.Text = 'Train Model';

            % Create LoadTestImageButton
            app.LoadTestImageButton = uibutton(app.MainPanel, 'push');
            app.LoadTestImageButton.ButtonPushedFcn = createCallbackFcn(app, @LoadTestImageButtonPushed, true);
            app.LoadTestImageButton.Position = [25 68 103 22];
            app.LoadTestImageButton.Text = 'Load Test Image';

            % Create ResetModelButton
            app.ResetModelButton = uibutton(app.MainPanel, 'push');
            app.ResetModelButton.ButtonPushedFcn = createCallbackFcn(app, @ResetModelButtonPushed, true);
            app.ResetModelButton.Position = [26 28 100 22];
            app.ResetModelButton.Text = 'Reset Model';

            % Create ImageAxesPanel
            app.ImageAxesPanel = uipanel(app.UIFigure);
            app.ImageAxesPanel.Title = 'ImageAxes';
            app.ImageAxesPanel.Position = [178 196 260 221];

            % Create UIAxes
            app.UIAxes = uiaxes(app.ImageAxesPanel);
            app.UIAxes.XTick = [];
            app.UIAxes.XTickLabel = '';
            app.UIAxes.YTickLabel = '';
            app.UIAxes.ButtonDownFcn = createCallbackFcn(app, @UIAxesButtonDown, true);
            app.UIAxes.Position = [17 41 223 145];

            % Create StatusPanel_2
            app.StatusPanel_2 = uipanel(app.UIFigure);
            app.StatusPanel_2.Title = 'Status';
            app.StatusPanel_2.Position = [461 256 152 101];

            % Create StatusTextArea
            app.StatusTextArea = uitextarea(app.StatusPanel_2);
            app.StatusTextArea.Position = [23 21 105 43];

            % Create ResultPanel
            app.ResultPanel = uipanel(app.UIFigure);
            app.ResultPanel.Title = 'Result';
            app.ResultPanel.Position = [178 75 260 103];

            % Create ResultTextArea
            app.ResultTextArea = uitextarea(app.ResultPanel);
            app.ResultTextArea.Position = [17 9 223 60];

            % Create BrainTumorDetectionSystemLabel
            app.BrainTumorDetectionSystemLabel = uilabel(app.UIFigure);
            app.BrainTumorDetectionSystemLabel.FontSize = 18;
            app.BrainTumorDetectionSystemLabel.FontWeight = 'bold';
            app.BrainTumorDetectionSystemLabel.Position = [215 434 270 26];
            app.BrainTumorDetectionSystemLabel.Text = 'Brain Tumor Detection System';

            % Create NameFaiadFaisalSarthokID220021116Label
            app.NameFaiadFaisalSarthokID220021116Label = uilabel(app.UIFigure);
            app.NameFaiadFaisalSarthokID220021116Label.Position = [462 85 154 30];
            app.NameFaiadFaisalSarthokID220021116Label.Text = {'Name: Faiad Faisal Sarthok'; 'ID: 220021116'};

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = BrainTumorDetectionAppSVM_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end