% Data root (laptop)
data_root = 'C:\DATA\vbshare\physionet_data'; % Starting path

datafolder = uigetdir(data_root, 'physioNet data folder') ;
data_root = datafolder;
%datafolder = 'C:\DATA\vbshare\bwekg_Data\physioNet\sample2017\validation';

% HDF5 file name
h5file = 'physio.h5';
h5path = [data_root, '\', h5file];

% Get the names of all data files
files = dir([datafolder, '\*.mat']);
fprintf(['Files found: ', num2str(length(files)), '\n']);

%% Load next record

fprintf('Processing files.\n')

for f = 1:length(files)
    
    %f = 1; % Start with the first one
    fname = files(f).name;
    [p, rname, x] = fileparts(fname);
    recordName = [datafolder, '\', rname];
    [tm, signal, Fs, siginfo] = rdmat(recordName);

    % Prepare ecg data to write in hdf5 file
    ecgdata = [signal, transpose(tm)]; % Time and signal columns
    size_ecgdata = size(ecgdata);
    code = ['SIGNAL'; 'TIME_S'];
    dpath_ecgdata = ['/', rname, '/ecgdata'];

    % Write ecg data
    h5create(h5path, dpath_ecgdata,  [size_ecgdata(2), size_ecgdata(1)])
    h5write(h5path, dpath_ecgdata, transpose(ecgdata))

    % Write column names as attributes
    h5writeatt(h5path, dpath_ecgdata, 'colnames', '[signal, time_s]')
    h5writeatt(h5path, dpath_ecgdata, 'units', siginfo.Units)
    h5writeatt(h5path, dpath_ecgdata, 'baseline', siginfo.Baseline)
    h5writeatt(h5path, dpath_ecgdata, 'gain', siginfo.Gain)
    h5writeatt(h5path, dpath_ecgdata, 'description', siginfo.Description)
    h5writeatt(h5path, dpath_ecgdata, 'fmt', siginfo.fmt)
    h5writeatt(h5path, dpath_ecgdata, 'sampling_frequency', Fs)
    
    if (mod(f, 50) == 0)
        
        % Show progress every 500 files
        fprintf(['Completed: ', num2str(f), ' / ', num2str(length(files)), '\n'])
    
    end
    
end