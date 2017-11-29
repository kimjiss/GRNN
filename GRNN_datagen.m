clear all; clc;
shapeInserterGRE = vision.ShapeInserter('Shape','Circles','BorderColor','Custom',...
            'CustomBorderColor',uint8([0, 255, 0]));
        
for index = [5, 6, 7, 8, 9]
    video_path = sprintf('/media/jisu/UUI/human36m/S%d/Videos', index);
    seg_path = sprintf('/media/jisu/UUI/human36m/S%d/MySegmentsMat/ground_truth_bs', index);
    pose_path = sprintf('/media/jisu/UUI/human36m/S%d/MyPoseFeatures/D2_Positions', index);
    video_save_path = sprintf('./data/S%d/128x128Images', index);
    seg_save_path = sprintf('./data/S%d/128x128seg', index);
    pose_save_path = sprintf('./data/S%d/128x128Poses', index);
    video_format = '*.mp4';
    seg_format = '*.mat';
    pose_format = '*.cdf';
    
    video_list = dir(fullfile(video_path, video_format));
    seg_list = dir(fullfile(seg_path, seg_format));
    pose_list = dir(fullfile(pose_path, pose_format));
    
    for numfile = 1:length(pose_list)
        % LoadPose
        pose = cdfread(fullfile(pose_path, pose_list(numfile).name));
        pose = pose{1};
        pose = double(pose);
        % Load Segment mask
        load(fullfile(seg_path, seg_list(numfile).name));
        % Load Video
        vidObj = VideoReader(fullfile(video_path, video_list(numfile).name));
        vidHeight = vidObj.Height;
        vidWidth = vidObj.Width;
        s = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'));
        
        k = 1;
        iter = 0;
        while hasFrame(vidObj)
            s(k).cdata = readFrame(vidObj);
            
            if k > 1999
                % exception handling
                pose_len = length(pose);
                frame_len = length(s);
                len = min(pose_len, frame_len);

                % Make Image folder
                img_folder = strrep(video_list(numfile).name, '.mp4', '');
                mkdir(video_save_path, img_folder);
                
                % Make Seg folder
                img_folder = strrep(seg_list(numfile).name, '.mat', '');
                mkdir(seg_save_path, img_folder);
                
                % Make Pose folder
                img_folder = strrep(pose_list(numfile).name, '.cdf', '');
                mkdir(pose_save_path, img_folder);
                
                for numframe = 1 : len
                    img = s(numframe).cdata;
                    joint = pose(numframe + iter * 2000, :);
                    img = imresize(img, 0.256);
                    mask = Masks{numframe + iter * 2000};
                    mask = imresize(mask, 0.256);
                    %% Crop joint 
                    for ii = 1:32
                        joint(1,2*ii-1)= joint(1,2*ii-1) * 0.256;
                        joint(1,2*ii) = joint(1,2*ii) * 0.256;
%                         circle = int32([joint(1, 2*ii-1), joint(1, 2*ii), 3]);
%                         img = shapeInserterGRE(img, circle);
                    end
%                     imshow(img);
                    %% Save cropped image
                    image_name = sprintf('%04d', numframe + iter * 2000);
                    save_path = fullfile(video_save_path, img_folder);

                    fn = sprintf('%s/%04d.jpg', save_path, numframe + iter * 2000);
                    imwrite(img, fn);
                    
                    %% Save cropped seg
                    seg_name = sprintf('%04d', numframe + iter * 2000);
                    save_path = fullfile(seg_save_path, img_folder);
                    fn = sprintf('%s/%04d.png', save_path, numframe + iter * 2000);
                    imwrite(mask, fn);
                    
                    %% Save cropped joint
                    txt_name = sprintf('%04d', numframe + iter * 2000);
                    save_path = fullfile(pose_save_path, img_folder);
                    fn = sprintf('%s/%04d.txt', save_path, numframe + iter * 2000);
                    fileID = fopen(fn, 'w');
                    fprintf(fileID, '%d ', joint(1, :));
                    fclose(fileID);

                end
                iter = iter + 1;
                k = 0;
                clear s;
            end
            k = k+1;                
        end
        % exception handling
        pose_len = length(pose) - iter * 2000;
        frame_len = length(s);
        len = min(pose_len, frame_len);
        %% Make Image folder
        img_folder = strrep(video_list(numfile).name, '.mp4', '');
        mkdir(video_save_path, img_folder);
        
        %% Make Seg folder
        img_folder = strrep(seg_list(numfile).name, '.mat', '');
        mkdir(seg_save_path, img_folder);
                
        %% Make Pose folder
        img_folder = strrep(pose_list(numfile).name, '.cdf', '');
        mkdir(pose_save_path, img_folder);
        
        for numframe = 1 : len

            img = s(numframe).cdata;
            joint = pose(numframe + iter * 2000, :);
            img = imresize(img, 0.256);
            mask = Masks{numframe + iter * 2000};
            mask = imresize(mask, 0.256);
            %% Crop joint 
            for ii = 1:32
                joint(1,2*ii-1)= joint(1,2*ii-1) * 0.256;
                joint(1,2*ii) = joint(1,2*ii) * 0.256;
%                 circle = int32([joint(1, 2*ii-1), joint(1, 2*ii), 3]);
%                 img = shapeInserterGRE(img, circle);
            end
%             imshow(img);
            %% Save cropped image
            image_name = sprintf('%04d', numframe + iter * 2000);
            save_path = fullfile(video_save_path, img_folder);

            fn = sprintf('%s/%04d.jpg', save_path, numframe + iter * 2000);
            imwrite(img, fn);
            
            %% Save cropped seg
            seg_name = sprintf('%04d', numframe + iter * 2000);
            save_path = fullfile(seg_save_path, img_folder);
            fn = sprintf('%s/%04d.png', save_path, numframe + iter * 2000);
            imwrite(mask, fn);
            
            %% Save cropped joint
            txt_name = sprintf('%04d', numframe + iter * 2000);
            save_path = fullfile(pose_save_path, img_folder);
            fn = sprintf('%s/%04d.txt', save_path, numframe + iter * 2000);
            fileID = fopen(fn, 'w');
            fprintf(fileID, '%d ', joint(1, :));
            fclose(fileID);
        end
        fprintf('S%d: %d-th video complete\n', index, numfile);
        
        
        
        
    end
    
end














