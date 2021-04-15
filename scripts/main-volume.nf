nextflow.enable.dsl = 2

include { scrubImage } from "./cleaning.nf"

params.bin = "${workflow.scriptFile.getParent()}/bin/"
usage = file("${workflow.scriptFile.getParent()}/usage")
printhelp = params.help


req_param = ["--fmriprep": "$params.fmriprep",
             "--cleanconf": "$params.cleanconf",
             "--outputdir":"$params.outputdir"]

missing_req = req_param.grep{ (it.value == null || it.value == "") }

if (missing_req) {
    log.error("Missing required command-line argument(s)!")
    missing_req.each{ log.error("Missing ${it.key}") }
    printhelp = true
}

if (printhelp) {
    print(usage.text)
    System.exit(0)
}

log.info("fMRIPrep directory: $params.fmriprep")
log.info("Output directory: $params.outputdir")
base_entities = /.+(?=_desc)/

// Retrieve all available fMRI files, store with base entities
fmri_scans = Channel.fromPath("$params.fmriprep/sub-*/ses-*/func/" +
                              "*desc-preproc_bold.nii.gz")
                    .map{p -> p.getBaseName()}
                    .map{p -> [(p =~ base_entities)[0], p]}

// Filter for required bids entities
if (params.bids_entities){
    log.info("Using the following bids entities")
    search_items = params.bids_entities.collect{e -> "${e.key}-${e.value}".toString()}
    search_items.each{ e -> log.info("$e") }
    fmri_scans = fmri_scans.filter{ entities, _ ->
        entities.split("_").toList().containsAll(search_items)
    }
}



workflow{

    main:

    // Perform volume scrubbing
    i_scrubImage = fmri_scans.map{ e, f ->
        [
            e,f,
            f.toString()
             .replaceAll("_desc-preproc_bold.nii", "_desc-confounds_timeseries.tsv")
        ]
    }
    scrubImage(i_scrubImage)
}
