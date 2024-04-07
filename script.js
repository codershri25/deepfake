function validateSize(input) {
    const fileSize = input.files[0].size / 1024 / 1024; // in MiB
    if (fileSize > 10) {
        alert('File size exceeds 10 MiB');
        // $(file).val(''); //for clearing with Jquery
    } else {
        // Proceed further
    }
}