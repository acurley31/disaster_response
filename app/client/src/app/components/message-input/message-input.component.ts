import { Component, OnInit, OnChanges, Input, Output, EventEmitter } from '@angular/core';
import { FormBuilder, Validators } from "@angular/forms";

@Component({
  selector: 'app-message-input',
  templateUrl: './message-input.component.html',
  styleUrls: ['./message-input.component.scss']
})
export class MessageInputComponent implements OnInit, OnChanges {
    @Input() message;
    @Input() isLoading: boolean = false;
    @Output() classify = new EventEmitter();
    form;

    // Constructor
    constructor(private fb: FormBuilder) { 
        
        // Initialize the form
        this.form = this.fb.group({
            message: ["", Validators.required],
        })
    
    }


    // On initializaton
    ngOnInit() {
    }


    // On changes
    ngOnChanges() {
        if (this.message) {
            this.form.patchValue({ message: this.message.message })
        }
    }


    // Return the list of message classifications
/*    get classifications() {
        
        // Check for a valid message
        if (!this.message) {
            return []
        }

        // Otherwise, extract and return the classifications
        let classifications = [];
        for (var prop in this.message) {
            if (prop !== "id" && prop !== "message") {
                classifications.push({ label: prop, value: this.message[prop] })
            }
        
        }

        return classifications

    }
*/

    // On classify
    onClassify(event) {
        const data = {...this.form.value}
        this.classify.emit(data)
    }

}
