nextflow.enable.dsl = 2

include { scrubImage; deriveConnectivity } from "./modules.nf"

usage = file("${workflow.scriptFile.getParent()}/usage")
printhelp = params.help


req_param = ["--fmriprep": "$params.fmriprep",
             "--cleanconf": "$params.cleanconf",
             "--parcellation": "$params.parcellation",
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

log.info("Project directory: ${workflow.projectDir}")
log.info("fMRIPrep directory: $params.fmriprep")
log.info("Output directory: $params.outputdir")
base_entities = /.+(?=_desc)/

// Retrieve all available fMRI files, store with base entities
fmri_scans = Channel.fromPath("$params.fmriprep/sub-*/ses-*/func/" +
                              "*desc-preproc_bold.nii.gz")
                    .map{p -> [(p.getBaseName() =~ base_entities)[0], p]}

// Filter for required bids entities
if (params.bids_entities){
    log.info("Using the following bids entities")
    search_items = params.bids_entities.collect{e -> "${e.key}-${e.value}".toString()}
    search_items.each{ e -> log.info("$e") }
    fmri_scans = fmri_scans.filter{ entities, _ ->
        entities.split("_").toList().containsAll(search_items)
    }
}

// Filter for subjects
if (params.subjects){
    log.info("Subjects list provided: $params.subjects")
    subjects_channel = Channel.fromPath(params.subjects)
                        .splitText(){it.strip()}
    fmri_scans = fmri_scans.map{e, f ->
        [
            (e =~ /sub-[A-Za-z0-9]+/)[0], e, f
        ]
    }
    .join(subjects_channel)
    .map{s,e,f -> [e,f]}

}

workflow{

    main:

    i_scrubImage = fmri_scans.map{ e, f ->
        [
            e,f,
            f.toString()
             .replaceAll("_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv")
        ]
    }
    .map{e,f,c -> [e, f, c - ~/_space-[A-Za-z0-9]+/]}
    .combine(params.METHODS)
    scrubImage(i_scrubImage)
    
    i_connectivity = scrubImage.out.clean_img.map{e,m,f ->
        [e,m,f,params.parcellation]
    }
    deriveConnectivity(i_connectivity)
}
